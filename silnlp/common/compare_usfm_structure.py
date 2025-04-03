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
    sent: str, stylesheet: UsfmStylesheet = UsfmStylesheet("usfm.sty"), to_ignore: List[str] = []
) -> Tuple[str, List[str]]:
    markers = []
    usfm_tokenizer = UsfmTokenizer(stylesheet)
    curr_embed = None
    filtered_sent = ""
    for tok in usfm_tokenizer.tokenize(sent):
        if curr_embed is not None:
            if tok.type == UsfmTokenType.END and tok.marker[:-1] == curr_embed.marker:
                curr_embed = None
        elif tok.type == UsfmTokenType.TEXT:
            filtered_sent += tok.to_usfm()
        elif tok.marker is None:
            continue
        elif tok.type == UsfmTokenType.NOTE or tok.marker in CHARACTER_TYPE_EMBEDS:
            if tok.end_marker is not None:
                curr_embed = tok
        elif tok.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHARACTER, UsfmTokenType.END]:
            if tok.marker not in to_ignore:
                filtered_sent += tok.to_usfm()
                markers.append(tok.marker)

    return filtered_sent, markers


# Assumes that the files have identical USFM structure
def evaluate_usfm_marker_placement(
    gold_book_path: Path, pred_book_path: Path, book: Optional[str] = None, to_ignore: Optional[List[str]] = []
) -> Tuple[float, float]:
    try:
        settings = FileParatextProjectSettingsParser(gold_book_path.parent).parse()
        stylesheet = settings.stylesheet
        gold_file_text = UsfmFileText(
            stylesheet,
            settings.encoding,
            settings.get_book_id(gold_book_path.name),
            gold_book_path,
            settings.versification,
            include_markers=True,
            include_all_text=True,
        )
        pred_file_text = UsfmFileText(
            settings.stylesheet,
            settings.encoding,
            settings.get_book_id(gold_book_path.name),
            pred_book_path,
            settings.versification,
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

    gold_sent_toks = []
    pred_sent_toks = []
    num_markers = []
    for gs, ps in zip(gold_file_text, pred_file_text):
        if len(gs.ref.path) > 0 and gs.ref.path[-1].name in PARAGRAPH_TYPE_EMBEDS:
            continue

        gs_text, gold_markers = filter_markers(gs.text, stylesheet, to_ignore)
        ps_text, pred_markers = filter_markers(ps.text, stylesheet, to_ignore)

        if len(gold_markers) == 0:
            continue
        num_markers.append(len(gold_markers))

        # Add markers that did not get read in to the prediction tokens
        # This only happens to paragraph markers that are placed at the end of a verse
        for marker in pred_markers:
            gold_markers.pop(gold_markers.index(marker))

        gold_sent_toks.append(list(tokenizer.tokenize(gs_text)))
        pred_sent_toks.append(list(tokenizer.tokenize(ps_text)) + gold_markers)

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

    avg_jaro, avg_dist = evaluate_usfm_marker_placement(Path(args.gold), Path(args.pred), args.book, to_ignore)

    LOGGER.info(f"Average (scaled) Jaro similarity of verses with placed markers: {avg_jaro}")
    LOGGER.info(f"Average Levenshtein distance per marker of verses with placed markers: {avg_dist}")


if __name__ == "__main__":
    main()
