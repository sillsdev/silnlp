import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from machine.corpora import (
    FileParatextProjectSettingsParser,
    ScriptureRef,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
    UsfmFileText,
    UsfmStylesheet,
    UsfmTextType,
    parse_usfm,
)
from machine.scripture import book_number_to_id, get_chapters
from transformers.trainer_utils import get_last_checkpoint

from ..common.paratext import book_file_name_digits, get_book_path, get_project_dir
from ..common.postprocesser import PostprocessConfig, PostprocessHandler
from ..common.usfm_utils import PARAGRAPH_TYPE_EMBEDS
from ..common.utils import get_git_revision_hash
from .clearml_connection import SILClearML
from .config import Config
from .config_utils import load_config
from .hugging_face_config import get_best_checkpoint

LOGGER = logging.getLogger(__package__ + ".postprocess")


# Takes the path to a USFM file and the relevant info to parse it
# and returns the text of all non-embed sentences and their respective references,
# along with any remarks (\rem) that were inserted at the beginning of the file
def get_sentences(
    book_path: Path, stylesheet: UsfmStylesheet, encoding: str, book: str, chapters: List[int] = []
) -> Tuple[List[str], List[ScriptureRef], List[str]]:
    sents = []
    refs = []
    draft_remarks = []
    for sent in UsfmFileText(stylesheet, encoding, book, book_path, include_all_text=True):
        marker = sent.ref.path[-1].name if len(sent.ref.path) > 0 else ""
        if marker == "rem" and len(refs) == 0:
            draft_remarks.append(sent.text)
            continue
        if (
            marker in PARAGRAPH_TYPE_EMBEDS
            or stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT
            or (len(chapters) > 0 and sent.ref.chapter_num not in chapters)
        ):
            continue

        sents.append(re.sub(" +", " ", sent.text.strip()))
        refs.append(sent.ref)

    return sents, refs, draft_remarks


# Get the paths of all drafts that would be produced by an experiment's translate config and that exist
def get_draft_paths_from_exp(config: Config) -> Tuple[List[Path], List[Path], List[PostprocessConfig]]:
    with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
        translate_requests = yaml.safe_load(file).get("translate", [])

    src_paths = []
    draft_paths = []
    postprocess_configs = []
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
                src_paths.append(src_path)
                draft_paths.append(draft_path)
                postprocess_configs.append(postprocess_config)
            elif draft_path.with_suffix(f".{1}{draft_path.suffix}").exists():  # multiple drafts
                for i in range(1, config.infer.get("num_drafts", 1) + 1):
                    src_paths.append(src_path)
                    draft_paths.append(draft_path.with_suffix(f".{i}{draft_path.suffix}"))
                    postprocess_configs.append(postprocess_config)
            else:
                LOGGER.warning(f"Draft not found: {draft_path}")

    return src_paths, draft_paths, postprocess_configs


def postprocess_draft(
    src_path: Path,
    draft_path: Path,
    postprocess_handler: PostprocessHandler,
    book: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> None:
    if str(src_path).startswith(str(get_project_dir(""))):
        settings = FileParatextProjectSettingsParser(src_path.parent).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        book = settings.get_book_id(src_path.name)
    else:
        stylesheet = UsfmStylesheet("usfm.sty")
        encoding = "utf-8-sig"

    src_sents, src_refs, _ = get_sentences(src_path, stylesheet, encoding, book)
    draft_sents, draft_refs, draft_remarks = get_sentences(draft_path, stylesheet, encoding, book)

    # Verify reference parity
    if len(src_refs) != len(draft_refs):
        LOGGER.warning(f"Can't process {src_path} and {draft_path}: Unequal number of verses/references")
        return
    for src_ref, draft_ref in zip(src_refs, draft_refs):
        if src_ref.to_relaxed() != draft_ref.to_relaxed():
            LOGGER.warning(
                f"Can't process {src_path} and {draft_path}: Mismatched ref, {src_ref} != {draft_ref}. Files must have the exact same USFM structure"
            )
            return

    postprocess_handler.construct_rows(src_refs, src_sents, draft_sents)

    with src_path.open(encoding=encoding) as f:
        usfm = f.read()

    for config in postprocess_handler.configs:
        handler = UpdateUsfmParserHandler(
            rows=config.rows,
            id_text=book,
            text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
            paragraph_behavior=config.get_paragraph_behavior(),
            embed_behavior=config.get_embed_behavior(),
            style_behavior=config.get_style_behavior(),
            update_block_handlers=config.update_block_handlers,
            remarks=(draft_remarks + [config.get_postprocess_remark()]),
        )
        parse_usfm(usfm, handler)
        usfm_out = handler.get_usfm()

        if not out_dir:
            out_dir = draft_path.parent
        out_path = out_dir / f"{draft_path.stem}{config.get_postprocess_suffix()}{draft_path.suffix}"
        with out_path.open("w", encoding="utf-8" if encoding == "utf-8-sig" else encoding) as f:
            f.write(usfm_out)


def postprocess_experiment(config: Config, out_dir: Optional[Path] = None) -> None:
    src_paths, draft_paths, legacy_pcs = get_draft_paths_from_exp(config)
    with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
        postprocess_configs = yaml.safe_load(file).get("postprocess", [])

    postprocess_handler = PostprocessHandler([PostprocessConfig(pc) for pc in postprocess_configs], include_base=False)

    for src_path, draft_path, legacy_pc in zip(src_paths, draft_paths, legacy_pcs):
        if postprocess_configs:
            postprocess_draft(src_path, draft_path, postprocess_handler, out_dir=out_dir)
        elif not legacy_pc.is_base_config():
            postprocess_draft(src_path, draft_path, PostprocessHandler([legacy_pc], False), out_dir=out_dir)


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
