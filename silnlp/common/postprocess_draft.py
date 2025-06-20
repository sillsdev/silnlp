import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple

import yaml
from machine.corpora import (
    FileParatextProjectSettingsParser,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
    UsfmFileText,
    UsfmStylesheet,
    UsfmTextType,
    parse_usfm,
)
from machine.scripture import book_id_to_number
from transformers.trainer_utils import get_last_checkpoint

from ..nmt.clearml_connection import SILClearML
from ..nmt.config import Config
from ..nmt.config_utils import create_config
from ..nmt.hugging_face_config import get_best_checkpoint
from .paratext import book_file_name_digits, get_book_path, get_project_dir
from .usfm_preservation import PARAGRAPH_TYPE_EMBEDS, construct_place_markers_handler
from .utils import get_mt_exp_dir

LOGGER = logging.getLogger(__package__ + ".postprocess_draft")


# NOTE: only using first book of first translate request for now
def get_paths_from_exp(config: Config) -> Tuple[Path, Path]:
    # TODO: default to first draft in the infer folder
    if not (config.exp_dir / "translate_config.yml").exists():
        raise ValueError("Experiment translate_config.yml not found. Please use --source and --draft options instead.")

    with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
        translate_config = yaml.safe_load(file)["translate"][0]
    src_project = translate_config.get("src_project", next(iter(config.src_projects)))
    books = translate_config["books"]
    book = books[0][:3] if isinstance(books, list) else books.split(";")[0][:3]
    book_num = book_id_to_number(book)

    ckpt = translate_config.get("checkpoint", "last")
    if ckpt == "best":
        step_str = get_best_checkpoint(config.model_dir).name[11:]
    elif ckpt == "last":
        step_str = Path(get_last_checkpoint(config.model_dir)).name[11:]
    else:
        step_str = str(ckpt)

    return (
        get_book_path(src_project, book),
        config.exp_dir / "infer" / step_str / src_project / f"{book_file_name_digits(book_num)}{book}.SFM",
    )


def insert_draft_remarks(usfm: str, remarks: List[str]) -> str:
    lines = usfm.split("\n")
    remark_lines = [f"\\rem {r}" for r in remarks]
    return "\n".join(lines[:1] + remark_lines + lines[1:])


def get_sentences(
    book_path: Path, stylesheet: UsfmStylesheet, encoding: str, book: str, chapters: List[int] = []
) -> Tuple[List[str], List[ScriptureRef], List[str]]:
    sents = []
    refs = []
    draft_remarks = []
    for sent in UsfmFileText(stylesheet, encoding, book, book_path, include_all_text=True):
        marker = sent.ref.path[-1].name if len(sent.ref.path) > 0 else ""
        if marker == "rem" and len(refs) == 0:  # TODO: \ide and \usfm lines could potentially come before the remark(s)
            draft_remarks.append(sent.text)
            continue
        if (
            marker in PARAGRAPH_TYPE_EMBEDS
            or stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT
            # or len(sent.text.strip()) == 0
            or (len(chapters) > 0 and sent.ref.chapter_num not in chapters)
        ):
            continue

        sents.append(re.sub(" +", " ", sent.text.strip()))
        refs.append(sent.ref)

    return sents, refs, draft_remarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Applies draft postprocessing steps to a draft.")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Name of an experiment directory in MT/experiments. \
        If this option is used, the experiment's translate config will be used to find source and draft files.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Path of the source USFM file. \
        If in a Paratext project, the project settings will be used when reading the files.",
    )
    parser.add_argument(
        "--draft",
        default=None,
        help="Path of the draft USFM file that postprocessing will be applied to. \
        Must have the exact same USFM structure as 'source', which it will if it is a draft from that source.",
    )
    parser.add_argument(
        "--book",
        default=None,
        help="3-letter book id of book being evaluated, e.g. MAT. \
        Only necessary if the source file is not in a Paratext project directory.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Output folder for the postprocessed draft. Defaults to the folder of the original draft.",
    )
    parser.add_argument(
        "--include-paragraph-markers",
        default=False,
        action="store_true",
        help="Attempt to place paragraph markers in translated verses based on the source project's markers",
    )
    parser.add_argument(
        "--include-style-markers",
        default=False,
        action="store_true",
        help="Attempt to place style markers in translated verses based on the source project's markers",
    )
    parser.add_argument(
        "--include-embeds",
        default=False,
        action="store_true",
        help="Carry over embeds from the source project to the output without translating them",
    )
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    args = parser.parse_args()

    if args.experiment and (args.source or args.draft or args.book):
        LOGGER.info("--experiment option used. --source, --draft, and --book will be ignored.")
    if not (args.experiment or (args.source and args.draft)):
        raise ValueError("Not enough options used. Please use --experiment OR --source and --draft.")

    experiment = args.experiment.replace("\\", "/") if args.experiment else None
    if experiment and get_mt_exp_dir(experiment).exists():
        exp_dir = get_mt_exp_dir(experiment)
        if args.clearml_queue is not None:
            if "cpu" not in args.clearml_queue:
                raise ValueError("Running this script on a GPU queue will not speed it up. Please only use CPU queues.")
            clearml = SILClearML(experiment, args.clearml_queue)
            config = clearml.config
        else:
            with (exp_dir / "config.yml").open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                config = create_config(exp_dir, config)

        src_path, draft_path = get_paths_from_exp(config)
    elif args.clearml_queue is not None:
        raise ValueError("Must use --experiment option to use ClearML.")
    else:
        src_path = Path(args.source.replace("\\", "/"))
        draft_path = Path(args.draft.replace("\\", "/"))

    # If no postprocessing options are used, use any postprocessing requests in the experiment's translate config
    if args.include_paragraph_markers or args.include_style_markers or args.include_embeds:
        postprocess_configs = [
            {
                "include_paragraph_markers": args.include_paragraph_markers,
                "include_style_markers": args.include_style_markers,
                "include_embeds": args.include_embeds,
            }
        ]
    else:
        if args.experiment:
            LOGGER.info("No postprocessing options used. Applying postprocessing requests from translate config.")
            with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
                postprocess_configs = yaml.safe_load(file).get("postprocess", [])
            if len(postprocess_configs) == 0:
                LOGGER.info("No postprocessing requests found.")
                exit()
        else:
            LOGGER.info("Please use at least one postprocessing option.")
            exit()

    if str(src_path).startswith(str(get_project_dir(""))):
        settings = FileParatextProjectSettingsParser(src_path.parent).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        book = settings.get_book_id(src_path.name)
    else:
        stylesheet = UsfmStylesheet("usfm.sty")
        encoding = "utf-8-sig"
        book = args.book
        if book is None:
            raise ValueError(
                "--book argument must be passed if the source file is not in a Paratext project directory."
            )

    src_sents, src_refs, _ = get_sentences(src_path, stylesheet, encoding, book)
    draft_sents, draft_refs, draft_remarks = get_sentences(draft_path, stylesheet, encoding, book)

    if len(src_refs) != len(draft_refs):
        raise ValueError("Different number of verses/references between source and draft.")
    for src_ref, draft_ref in zip(src_refs, draft_refs):
        if src_ref.to_relaxed() != draft_ref.to_relaxed():
            raise ValueError(
                f"'source' and 'draft' must have the exact same USFM structure. Mismatched ref: {src_ref} {draft_ref}"
            )

    if any(
        ppc.get("include_paragraph_markers", False) or ppc.get("include_style_markers", False)
        for ppc in postprocess_configs
    ):
        place_markers_handler = construct_place_markers_handler(src_refs, src_sents, draft_sents)

    for postprocess_config in postprocess_configs:
        update_block_handlers = []
        if postprocess_config.get("include_paragraph_markers", False) or postprocess_config.get(
            "include_style_markers", False
        ):
            update_block_handlers.append(place_markers_handler)

        paragraph_behavior = (
            UpdateUsfmMarkerBehavior.PRESERVE
            if postprocess_config.get("include_paragraph_markers", False)
            else UpdateUsfmMarkerBehavior.STRIP
        )
        style_behavior = (
            UpdateUsfmMarkerBehavior.PRESERVE
            if postprocess_config.get("include_style_markers", False)
            else UpdateUsfmMarkerBehavior.STRIP
        )
        embed_behavior = (
            UpdateUsfmMarkerBehavior.PRESERVE
            if postprocess_config.get("include_embeds", False)
            else UpdateUsfmMarkerBehavior.STRIP
        )
        marker_placement_suffix = (
            "_"
            + ("p" if postprocess_config.get("include_paragraph_markers", False) else "")
            + ("s" if postprocess_config.get("include_style_markers", False) else "")
            + ("e" if postprocess_config.get("include_embeds", False) else "")
        )

        with src_path.open(encoding=encoding) as f:
            usfm = f.read()
        handler = UpdateUsfmParserHandler(
            rows=[([ref], sent) for ref, sent in zip(src_refs, draft_sents)],
            id_text=book,
            text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
            paragraph_behavior=paragraph_behavior,
            embed_behavior=embed_behavior,
            style_behavior=style_behavior,
            update_block_handlers=update_block_handlers,
        )
        parse_usfm(usfm, handler)
        usfm_out = handler.get_usfm()

        usfm_out = insert_draft_remarks(usfm_out, draft_remarks)

        out_dir = Path(args.output_folder.replace("\\", "/")) if args.output_folder else draft_path.parent
        out_path = out_dir / f"{draft_path.stem}{marker_placement_suffix}{draft_path.suffix}"
        with out_path.open("w", encoding="utf-8" if encoding == "utf-8-sig" else encoding) as f:
            f.write(usfm_out)


if __name__ == "__main__":
    main()
