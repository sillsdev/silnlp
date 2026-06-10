import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from attr import dataclass
from machine.corpora import (
    FileParatextProjectSettingsParser,
    ScriptureRef,
    UsfmFileText,
    UsfmStylesheet,
    UsfmTextType,
    UsfmToken,
    UsfmTokenizer,
    UsfmTokenType,
)
from machine.scripture import book_number_to_id, get_chapters
from transformers.trainer_utils import get_last_checkpoint

from ..common.environment import SilNlpEnv
from ..common.paratext import book_file_name_digits, get_book_path
from ..common.postprocesser import (
    NoDetectedQuoteConventionException,
    PostprocessConfig,
    PostprocessHandler,
    UnknownQuoteConventionException,
)
from ..common.usfm_utils import PARAGRAPH_TYPE_EMBEDS
from ..common.utils import get_git_revision_hash
from .clearml_connection import TAGS_LIST, SILClearML
from .config import Config
from .config_utils import load_config
from .corpora import CorpusPair
from .hugging_face_config import get_best_checkpoint

LOGGER = logging.getLogger(__package__ + ".postprocess")


# Replicating machine.py's filter_tokens_by_chapter function here since it's not currently public
def filter_tokens_by_chapter(tokens: List[UsfmToken], chapters: List[int]) -> List[UsfmToken]:
    filtered: List[UsfmToken] = []
    in_chapter = False
    in_id_marker = False
    for index, token in enumerate(tokens):
        if index == 0 and token.marker == "id":
            in_id_marker = True
            if 1 in chapters:
                in_chapter = True
        elif in_id_marker and token.marker is not None and token.marker != "id":
            in_id_marker = False
        elif token.type == UsfmTokenType.CHAPTER:
            data = str(token.data).strip() if token.data is not None else ""
            in_chapter = data.isdigit() and int(data) in chapters
        if in_id_marker or in_chapter:
            filtered.append(token)
    return filtered


@dataclass
class Sentence:
    text: str
    ref: ScriptureRef


@dataclass
class DraftSentences:
    sentences: List[Sentence]
    remarks: List[Tuple[int, str]]


# Takes the path to a USFM file and the relevant info to parse it
# and returns the text of all non-embed sentences and their respective references,
# along with any remarks (\rem), paired with the chapter they appear in (0 for book-level remarks)
def get_sentences(
    book_path: Path, stylesheet: UsfmStylesheet, encoding: str, book: str, chapters: List[int] = []
) -> DraftSentences:
    draft_sentences = DraftSentences([], [])

    for sent in UsfmFileText(stylesheet, encoding, book, book_path, include_all_text=True):
        marker = sent.ref.path[-1].name if len(sent.ref.path) > 0 else ""
        if marker == "rem":
            draft_sentences.remarks.append((sent.ref.chapter_num, sent.text))
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
    source_project: str


# Get the paths of all drafts that would be produced by an experiment's translate config and that exist
def get_draft_paths_from_exp(config: Config, environment: SilNlpEnv) -> List[DraftMetadata]:
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

        book_nums = get_chapters(translate_request.get("books", [])).keys()
        for book_num in book_nums:
            book = book_number_to_id(book_num)

            src_path = get_book_path(src_project, book, environment)
            draft_path = (
                config.exp_dir / "infer" / step_str / src_project / f"{book_file_name_digits(book_num)}{book}.SFM"
            )
            if draft_path.exists():
                draft_metadata_list.append(
                    DraftMetadata(source_path=src_path, draft_path=draft_path, source_project=src_project)
                )
            elif draft_path.with_suffix(f".{1}{draft_path.suffix}").exists():  # multiple drafts
                for i in range(1, config.infer.get("num_drafts", 1) + 1):
                    draft_metadata_list.append(
                        DraftMetadata(
                            source_path=src_path,
                            draft_path=draft_path.with_suffix(f".{i}{draft_path.suffix}"),
                            source_project=src_project,
                        )
                    )
            else:
                LOGGER.warning(f"Draft not found: {draft_path}")

    return draft_metadata_list


def postprocess_draft(
    draft_metadata: DraftMetadata,
    postprocess_handler: PostprocessHandler,
    book: Optional[str] = None,
    out_dir: Optional[Path] = None,
    training_corpus_pairs: Optional[List[CorpusPair]] = None,
    environment: SilNlpEnv = SilNlpEnv.create_standard_environment(),
) -> None:
    if training_corpus_pairs is None:
        training_corpus_pairs = []

    if str(draft_metadata.source_path).startswith(str(environment.get_paratext_project_dir(""))):
        settings = FileParatextProjectSettingsParser(draft_metadata.source_path.parent).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        book = settings.get_book_id(draft_metadata.source_path.name)
    else:
        stylesheet = UsfmStylesheet("usfm.sty")
        encoding = "utf-8-sig"

    draft_sentences = get_sentences(draft_metadata.draft_path, stylesheet, encoding, book)
    draft_chapters = sorted({sentence.ref.chapter_num for sentence in draft_sentences.sentences})
    src_sentences = get_sentences(draft_metadata.source_path, stylesheet, encoding, book, draft_chapters)

    # Verify reference parity
    if len(src_sentences.sentences) != len(draft_sentences.sentences):
        LOGGER.warning(
            f"Can't process {draft_metadata.source_path} and {draft_metadata.draft_path}: "
            f"Unequal number of verses/references"
        )
        return
    for src_sentence, draft_sentence in zip(src_sentences.sentences, draft_sentences.sentences):
        if src_sentence.ref.to_relaxed() != draft_sentence.ref.to_relaxed():
            LOGGER.warning(
                f"Can't process {draft_metadata.source_path} and {draft_metadata.draft_path}: "
                f"Mismatched ref, {src_sentence.ref} != {draft_sentence.ref}. "
                f"Files must have the exact same USFM structure"
            )
            return

    source_usfm = None
    if any(config.is_marker_processing_required() for config in postprocess_handler.configs):
        postprocess_handler.construct_rows(
            [s.ref for s in src_sentences.sentences],
            [s.text for s in src_sentences.sentences],
            [s.text for s in draft_sentences.sentences],
        )

        with draft_metadata.source_path.open(encoding=encoding) as f:
            source_usfm = f.read()
        if draft_chapters:
            tokenizer = UsfmTokenizer(stylesheet)
            source_usfm = tokenizer.detokenize(
                filter_tokens_by_chapter(tokenizer.tokenize(source_usfm), draft_chapters)
            )

    for config in postprocess_handler.configs:
        if config.is_marker_processing_required():
            place_markers_postprocessor = config.create_place_markers_postprocessor()
            remarks = [
                (chapter_num, place_markers_postprocessor.replace_paragraph_marker_remark(text))
                for chapter_num, text in draft_sentences.remarks
            ]
            target_usfm = place_markers_postprocessor.postprocess_usfm(
                source_usfm, config.rows, remarks, stylesheet=stylesheet
            )
        else:
            with draft_metadata.draft_path.open(encoding=encoding) as f:
                target_usfm = f.read()

        if config.is_quotation_mark_denormalization_required():
            try:
                quotation_denormalization_postprocessor = config.create_denormalize_quotation_marks_postprocessor(
                    training_corpus_pairs,
                )
                target_usfm = quotation_denormalization_postprocessor.postprocess_usfm(
                    target_usfm, stylesheet=stylesheet
                )
            except (UnknownQuoteConventionException, NoDetectedQuoteConventionException) as e:
                LOGGER.warning(str(e) + " Skipping quotation mark denormalization.")
                continue

        if not out_dir:
            out_dir = draft_metadata.draft_path.parent
        out_path = (
            out_dir
            / f"{draft_metadata.draft_path.stem}{config.get_postprocess_suffix()}{draft_metadata.draft_path.suffix}"
        )
        with out_path.open(
            "w", encoding="utf-8" if encoding == "utf-8-sig" or encoding == "utf_8_sig" else encoding
        ) as f:
            f.write(target_usfm)


def postprocess_experiment(
    config: Config,
    postprocess_handler: Optional[PostprocessHandler] = None,
    out_dir: Optional[Path] = None,
    environment: SilNlpEnv = SilNlpEnv.create_standard_environment(),
) -> None:
    draft_metadata_list = get_draft_paths_from_exp(config, environment)

    with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
        translate_config = yaml.safe_load(file)
        postprocess_configs = [PostprocessConfig(pc, environment) for pc in translate_config.get("postprocess", [])]

    if postprocess_handler is None:
        postprocess_handler = PostprocessHandler(postprocess_configs, include_base=False, environment=environment)

    for draft_metadata in draft_metadata_list:
        if postprocess_configs:
            postprocess_draft(
                draft_metadata,
                postprocess_handler,
                out_dir=out_dir,
                training_corpus_pairs=config.corpus_pairs,
                environment=environment,
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
    parser.add_argument(
        "--clearml-tag",
        metavar="tag",
        choices=TAGS_LIST,
        default=None,
        type=str,
        help=f"Tag to add to the ClearML Task - {TAGS_LIST}",
    )
    args = parser.parse_args()

    if args.clearml_queue is not None and args.clearml_tag is None:
        parser.error("Missing ClearML tag. Add a tag using --clearml-tag. Possible tags: " + f"{TAGS_LIST}")

    get_git_revision_hash()

    environment = SilNlpEnv.create_standard_environment()

    if args.clearml_queue is not None:
        clearml = SILClearML(args.experiment, args.clearml_queue, environment=environment)
        config = clearml.config
    else:
        config = load_config(args.experiment.replace("\\", "/"), environment)
    config.set_seed()

    postprocess_experiment(config, environment=environment)


if __name__ == "__main__":
    main()
