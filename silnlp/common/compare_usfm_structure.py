import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
from jarowinkler import jaro_similarity
from machine.corpora import (
    FileParatextProjectSettingsParser,
    UsfmFileText,
    UsfmStylesheet,
    UsfmTokenizer,
    UsfmTokenType,
)
from machine.tokenization import WhitespaceTokenizer

from .usfm_preservation import CHARACTER_TYPE_EMBEDS, PARAGRAPH_TYPE_EMBEDS

LOGGER = logging.getLogger(__package__ + ".compare_usfm_structure")


class WhitespaceMarkerTokenizer(WhitespaceTokenizer):
    def _is_whitespace(self, c: str) -> bool:
        return super()._is_whitespace(c) or c == "\\" or c == "*"


# Filter out embeds and ignored markers from sentence and create list of all remaining markers
def filter_markers(
    sent: str,
    stylesheet: UsfmStylesheet = UsfmStylesheet("usfm.sty"),
    only_paragraph: bool = False,
    only_style: bool = False,
    to_ignore: List[str] = [],
) -> Tuple[str, List[str]]:
    markers = []
    usfm_tokenizer = UsfmTokenizer(stylesheet)
    curr_embed = None
    filtered_sent = ""
    for tok in usfm_tokenizer.tokenize(sent):
        base_marker = tok.marker.strip("+*") if tok.marker is not None else None
        if curr_embed is not None:
            if tok.type == UsfmTokenType.END and base_marker == curr_embed:
                curr_embed = None
        elif tok.type == UsfmTokenType.TEXT:
            filtered_sent += tok.to_usfm()
        elif tok.marker is None:
            continue
        elif tok.type == UsfmTokenType.NOTE or base_marker in CHARACTER_TYPE_EMBEDS:
            if tok.end_marker is not None:
                curr_embed = base_marker
        elif (tok.type == UsfmTokenType.PARAGRAPH and not only_style) or (
            tok.type in [UsfmTokenType.CHARACTER, UsfmTokenType.END] and not only_paragraph
        ):
            if base_marker not in to_ignore:
                filtered_sent += tok.to_usfm()
                markers.append(tok.marker)

    return filtered_sent, markers


# Assumes that the files have identical USFM structure
def evaluate_usfm_marker_placement(
    gold_book_path: Path,
    pred_book_path: Path,
    book: Optional[str] = None,
    only_paragraph: bool = False,
    only_style: bool = False,
    to_ignore: List[str] = [],
) -> Optional[Tuple[float, float]]:
    try:
        settings = FileParatextProjectSettingsParser(gold_book_path.parent).parse()
        stylesheet = settings.stylesheet
        gold_file_text = UsfmFileText(
            stylesheet,
            settings.encoding,
            settings.get_book_id(gold_book_path.name),
            gold_book_path,
            include_markers=True,
            include_all_text=True,
        )
        pred_file_text = UsfmFileText(
            settings.stylesheet,
            settings.encoding,
            settings.get_book_id(gold_book_path.name),
            pred_book_path,
            include_markers=True,
            include_all_text=True,
        )
    except:
        if book is None:
            raise ValueError("--book argument must be passed if the gold file is not in a Paratext project directory.")

        stylesheet = UsfmStylesheet("usfm.sty")
        gold_file_text = UsfmFileText(
            stylesheet, "utf-8-sig", book, gold_book_path, include_markers=True, include_all_text=True
        )
        pred_file_text = UsfmFileText(
            stylesheet, "utf-8-sig", book, pred_book_path, include_markers=True, include_all_text=True
        )

    tokenizer = WhitespaceMarkerTokenizer()

    gold_file_sents = [
        gs.text for gs in gold_file_text if not (len(gs.ref.path) > 0 and gs.ref.path[-1].name in PARAGRAPH_TYPE_EMBEDS)
    ]
    pred_file_sents = [
        ps.text for ps in pred_file_text if not (len(ps.ref.path) > 0 and ps.ref.path[-1].name in PARAGRAPH_TYPE_EMBEDS)
    ]

    gold_sent_toks = []
    pred_sent_toks = []
    num_markers = []
    for gs, ps in zip(gold_file_sents, pred_file_sents):
        gs_text, gold_markers = filter_markers(gs, stylesheet, only_paragraph, only_style, to_ignore)
        ps_text, pred_markers = filter_markers(ps, stylesheet, only_paragraph, only_style, to_ignore)

        if len(gold_markers) == 0:
            continue
        num_markers.append(len(gold_markers))

        # Add markers that did not get read in to the prediction tokens
        # This only happens to paragraph markers that are placed at the end of a verse
        for marker in pred_markers:
            gold_markers.pop(gold_markers.index(marker))

        gold_sent_toks.append(list(tokenizer.tokenize(gs_text)))
        pred_sent_toks.append(list(tokenizer.tokenize(ps_text)) + gold_markers)

    # No verses with markers that should be evaluated
    if len(gold_sent_toks) == 0:
        return None

    jaro_scores = []
    dists_per_marker = []
    for gs, ps, n in zip(gold_sent_toks, pred_sent_toks, num_markers):
        # (scaled) Jaro similarity
        # The motivation for this altered metric is to show only the impact of marker placement on the similarity of the sentences,
        # since we're working under the assumption that all of the non-marker tokens match to each other trivially and inflate the base metric
        p_markers = n / len(gs)
        p_non_markers = 1 - p_markers
        jaro_scores.append((jaro_similarity(gs, ps) - p_non_markers) / p_markers)

        # Levenshtein distance
        dists_per_marker.append(nltk.edit_distance(gs, ps, transpositions=True) / n)

    return sum(jaro_scores) / len(gold_sent_toks), sum(dists_per_marker) / len(gold_sent_toks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compares USFM marker placement of the prediction file to the gold file."
    )
    parser.add_argument(
        "gold",
        help="Path of USFM file with correct marker placement. \
        If in a Paratext project, the project settings will be used when reading the files.",
    )
    parser.add_argument("pred", help="Path of USFM file to evaluate.")
    parser.add_argument(
        "--book",
        default=None,
        help="3-letter book id of book being evaluated, e.g. MAT. \
        Only necessary if the gold file is not in a Paratext project directory.",
    )
    parser.add_argument(
        "--ignored-markers",
        nargs="+",
        default=[],
        help="Paragraph and style markers to ignore for the evaluation. \
        Can be passed as multiple arguments or a single string with semi-colon-separated values.",
    )

    args = parser.parse_args()

    to_ignore = args.ignored_markers[0].split(";") if len(args.ignored_markers) == 1 else args.ignored_markers

    # Evaluate all marker placement
    scores = evaluate_usfm_marker_placement(Path(args.gold), Path(args.pred), args.book, to_ignore=to_ignore)
    if scores is None:
        LOGGER.info("No verses with markers found.")
        exit()
    LOGGER.info(f"Average (scaled) Jaro similarity of verses with placed markers: {scores[0]}")
    LOGGER.info(f"Average Levenshtein distance per marker of verses with placed markers: {scores[1]}")

    # Evaluate paragraph marker placement
    scores_para = evaluate_usfm_marker_placement(
        Path(args.gold), Path(args.pred), args.book, only_paragraph=True, to_ignore=to_ignore
    )
    if scores_para is None:
        LOGGER.info("No verses with paragraph markers found.")
        exit()

    # Evaluate style marker placement
    scores_style = evaluate_usfm_marker_placement(
        Path(args.gold), Path(args.pred), args.book, only_style=True, to_ignore=to_ignore
    )
    if scores_style is None:
        LOGGER.info("No verses with style markers found.")
        exit()

    LOGGER.info(
        f"Average (scaled) Jaro similarity of verses with placed markers (only paragraph markers): {scores_para[0]}"
    )
    LOGGER.info(
        f"Average Levenshtein distance per marker of verses with placed markers (only paragraph markers): {scores_para[1]}"
    )
    LOGGER.info(
        f"Average (scaled) Jaro similarity of verses with placed markers (only style markers): {scores_style[0]}"
    )
    LOGGER.info(
        f"Average Levenshtein distance per marker of verses with placed markers (only style markers): {scores_style[1]}"
    )


if __name__ == "__main__":
    main()
