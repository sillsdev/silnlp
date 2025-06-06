import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple

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

from .paratext import get_project_dir
from .usfm_preservation import PARAGRAPH_TYPE_EMBEDS, construct_place_markers_handler

LOGGER = logging.getLogger(__package__ + ".postprocess_draft")


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
    parser = argparse.ArgumentParser(
        description="Applies draft postprocessing steps to a draft. Can be used with no postprocessing options to create a base draft."
    )
    parser.add_argument(
        "source",
        help="Path of the source USFM file. \
        If in a Paratext project, the project settings will be used when reading the files.",
    )
    parser.add_argument(
        "draft",
        help="Path of the draft USFM file that postprocessing will be applied to. \
                        Must have the exact same USFM structure as 'source', which it will if it is a draft from that source.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Output folder for the postprocessed draft. Defaults to the folder of the original draft.",
    )
    parser.add_argument(
        "--book",
        default=None,
        help="3-letter book id of book being evaluated, e.g. MAT. \
        Only necessary if the source file is not in a Paratext project directory.",
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

    args = parser.parse_args()

    src_path = Path(args.source)
    draft_path = Path(args.draft)

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

    paragraph_behavior = (
        UpdateUsfmMarkerBehavior.PRESERVE if args.include_paragraph_markers else UpdateUsfmMarkerBehavior.STRIP
    )
    style_behavior = UpdateUsfmMarkerBehavior.PRESERVE if args.include_style_markers else UpdateUsfmMarkerBehavior.STRIP
    embed_behavior = UpdateUsfmMarkerBehavior.PRESERVE if args.include_embeds else UpdateUsfmMarkerBehavior.STRIP

    update_block_handlers = []
    if args.include_paragraph_markers or args.include_style_markers:
        update_block_handlers.append(construct_place_markers_handler(src_refs, src_sents, draft_sents))

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

    out_dir = Path(args.output_folder) if args.output_folder else draft_path.parent
    out_path = out_dir / f"{draft_path.stem}_postprocessed{draft_path.suffix}"
    with out_path.open("w", encoding="utf-8" if encoding == "utf-8-sig" else encoding) as f:
        f.write(usfm_out)


if __name__ == "__main__":
    main()
