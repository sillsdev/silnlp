import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional

import yaml
from attr import dataclass
from machine.corpora import FileParatextProjectSettingsParser, ScriptureRef, UsfmFileText, UsfmStylesheet, UsfmTextType
from machine.scripture import book_number_to_id, get_chapters
from transformers.trainer_utils import get_last_checkpoint

from ..common.paratext import book_file_name_digits, get_book_path, get_project_dir
from ..common.postprocesser import (
    NoDetectedQuoteConventionException,
    PostprocessConfig,
    PostprocessHandler,
    UnknownQuoteConventionException,
)
from ..common.usfm_utils import PARAGRAPH_TYPE_EMBEDS
from ..common.utils import get_git_revision_hash
from .clearml_connection import SILClearML
from .config import Config
from .config_utils import load_config
from .corpora import CorpusPair
from .hugging_face_config import get_best_checkpoint

LOGGER = logging.getLogger(__package__ + ".postprocess")


@dataclass
class Sentence:
    text: str
    ref: ScriptureRef


@dataclass
class DraftSentences:
    sentences: List[Sentence]
    remarks: List[str]


# Takes the path to a USFM file and the relevant info to parse it
# and returns the text of all non-embed sentences and their respective references,
# along with any remarks (\rem) that were inserted at the beginning of the file
def get_sentences(
    book_path: Path, stylesheet: UsfmStylesheet, encoding: str, book: str, chapters: List[int] = []
) -> DraftSentences:
    draft_sentences = DraftSentences([], [])

    for sent in UsfmFileText(stylesheet, encoding, book, book_path, include_all_text=True):
        marker = sent.ref.path[-1].name if len(sent.ref.path) > 0 else ""
        if marker == "rem" and len(draft_sentences.sentences) == 0:
            draft_sentences.remarks.append(sent.text)
            continue
        if (
            marker in PARAGRAPH_TYPE_EMBEDS
            or stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT
            or (len(chapters) > 0 and sent.ref.chapter_num not in chapters)
        ):
            continue

        draft_sentences.sentences.append(Sentence(re.sub(" +", " ", sent.text.strip()), sent.ref))

    return draft_sentences


@dataclass
class DraftMetadata:
    source_path: Path
    draft_path: Path
    postprocess_config: PostprocessConfig


# Get the paths of all drafts that would be produced by an experiment's translate config and that exist
def get_draft_paths_from_exp(config: Config) -> List[DraftMetadata]:
    with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as translate_config_file:
        translate_requests = yaml.safe_load(translate_config_file).get("translate", [])

    draft_metadata_list = []
    for translate_request in translate_requests:
        src_project = translate_request.get("src_project", next(iter(config.src_projects)))

        ckpt = translate_request.get("checkpoint", "last")
        if ckpt == "best":
            step_str = get_best_checkpoint(config.model_dir).name[11:]
        elif ckpt == "last":
            step_str = Path(get_last_checkpoint(config.model_dir)).name[11:]
        else:
            step_str = str(ckpt)

        # Backwards compatibility
        postprocess_config = PostprocessConfig(translate_request)

        book_nums = get_chapters(translate_request.get("books", [])).keys()
        for book_num in book_nums:
            book = book_number_to_id(book_num)

            src_path = get_book_path(src_project, book)
            draft_path = (
                config.exp_dir / "infer" / step_str / src_project / f"{book_file_name_digits(book_num)}{book}.SFM"
            )
            if draft_path.exists():
                draft_metadata_list.append(
                    DraftMetadata(
                        source_path=src_path,
                        draft_path=draft_path,
                        postprocess_config=postprocess_config,
                    )
                )
            elif draft_path.with_suffix(f".{1}{draft_path.suffix}").exists():  # multiple drafts
                for i in range(1, config.infer.get("num_drafts", 1) + 1):
                    draft_metadata_list.append(
                        DraftMetadata(
                            source_path=src_path,
                            draft_path=draft_path.with_suffix(f".{i}{draft_path.suffix}"),
                            postprocess_config=postprocess_config,
                        )
                    )
            else:
                LOGGER.warning(f"Draft not found: {draft_path}")

    return draft_metadata_list


def postprocess_draft(
    src_path: Path,
    draft_path: Path,
    postprocess_handler: PostprocessHandler,
    book: Optional[str] = None,
    out_dir: Optional[Path] = None,
    training_corpus_pairs: List[CorpusPair] = [],
) -> None:
    if str(src_path).startswith(str(get_project_dir(""))):
        settings = FileParatextProjectSettingsParser(src_path.parent).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        book = settings.get_book_id(src_path.name)
    else:
        stylesheet = UsfmStylesheet("usfm.sty")
        encoding = "utf-8-sig"

    src_sentences = get_sentences(src_path, stylesheet, encoding, book)
    draft_sentences = get_sentences(draft_path, stylesheet, encoding, book)

    # Verify reference parity
    if len(src_sentences.sentences) != len(draft_sentences.sentences):
        LOGGER.warning(f"Can't process {src_path} and {draft_path}: Unequal number of verses/references")
        return
    for src_sentence, draft_sentence in zip(src_sentences.sentences, draft_sentences.sentences):
        if src_sentence.ref.to_relaxed() != draft_sentence.ref.to_relaxed():
            LOGGER.warning(
                f"Can't process {src_path} and {draft_path}: Mismatched ref, {src_ref} != {draft_ref}. Files must have the exact same USFM structure"
            )
            return

    if any(config.is_marker_placement_required() for config in postprocess_handler.configs):
        postprocess_handler.construct_rows(
            [s.ref for s in src_sentences.sentences],
            [s.text for s in src_sentences.sentences],
            [s.text for s in draft_sentences.sentences],
        )

    with src_path.open(encoding=encoding) as f:
        source_usfm = f.read()

    for config in postprocess_handler.configs:
        if config.is_marker_placement_required():
            place_markers_postprocessor = config.create_place_markers_postprocessor()
            target_usfm = place_markers_postprocessor.postprocess_usfm(
                source_usfm, config.rows, draft_sentences.remarks
            )
        else:
            with draft_path.open(encoding=encoding) as f:
                target_usfm = f.read()

        if config.is_quotation_mark_denormalization_required():
            try:
                quotation_denormalization_postprocessor = config.create_denormalize_quotation_marks_postprocessor(
                    training_corpus_pairs
                )
                target_usfm = quotation_denormalization_postprocessor.postprocess_usfm(target_usfm)
            except (UnknownQuoteConventionException, NoDetectedQuoteConventionException) as e:
                raise e

        if not out_dir:
            out_dir = draft_path.parent
        out_path = out_dir / f"{draft_path.stem}{config.get_postprocess_suffix()}{draft_path.suffix}"
        with out_path.open(
            "w", encoding="utf-8" if encoding == "utf-8-sig" or encoding == "utf_8_sig" else encoding
        ) as f:
            f.write(target_usfm)


def postprocess_experiment(config: Config, out_dir: Optional[Path] = None) -> None:
    draft_metadata_list = get_draft_paths_from_exp(config)
    with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
        translate_config = yaml.safe_load(file)
        postprocess_configs = [PostprocessConfig(pc) for pc in translate_config.get("postprocess", [])]

    postprocess_handler = PostprocessHandler(postprocess_configs, include_base=False)

    for draft_metadata in draft_metadata_list:
        if postprocess_configs:
            postprocess_draft(
                draft_metadata.source_path,
                draft_metadata.draft_path,
                postprocess_handler,
                out_dir=out_dir,
                training_corpus_pairs=config.corpus_pairs,
            )
        elif not draft_metadata.postprocess_config.is_base_config():
            postprocess_draft(
                draft_metadata.source_path,
                draft_metadata.draft_path,
                PostprocessHandler([draft_metadata.postprocess_config], False),
                out_dir=out_dir,
                training_corpus_pairs=config.corpus_pairs,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postprocess the drafts created by an NMT model based on the experiment's translate config."
    )
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    args = parser.parse_args()

    get_git_revision_hash()

    if args.clearml_queue is not None:
        clearml = SILClearML(args.experiment, args.clearml_queue)
        config = clearml.config
    else:
        config = load_config(args.experiment.replace("\\", "/"))
    config.set_seed()

    postprocess_experiment(config)


if __name__ == "__main__":
    main()
