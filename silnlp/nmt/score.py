"""Score translation output produced outside silnlp,
using the same metrics and parameters as silnlp's test stage.
"""

import argparse
from pathlib import Path
from typing import List, Set

from .scoring_metrics import (
    CORPUS_SCORERS,
    DEFAULT_SACREBLEU_TOKENIZE,
    SENTENCE_SCORERS,
    PairScore,
    compute_corpus_scores,
    iter_verse_scores,
)

DEFAULT_SCORERS = {"bleu", "chrf3", "chrf3+", "chrf3++"}
VERSE_SCORES_SUFFIX = ".scores.csv"


def write_verse_scores(
    path: Path,
    pair_sys: List[str],
    pair_refs: List[List[str]],
    scorers: Set[str],
    sacrebleu_tokenize: str,
    ref_names: Set[str],
) -> None:
    scorers = scorers.intersection(SENTENCE_SCORERS)

    with open(path, "w", encoding="utf-8") as scores_file:
        for verse_score in iter_verse_scores(pair_sys, pair_refs, scorers, sacrebleu_tokenize):
            score = PairScore(
                book=str(verse_score.index + 1),
                src_iso="",
                trg_iso="",
                scores=verse_score.scores,
                sent_len=1,
                projects=ref_names,
            )

            if verse_score.index == 0:
                score.writeHeader(scores_file)

            score.write(scores_file)


def _read_all_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description=("Score translation output produced outside silnlp"))
    parser.add_argument("--hyp", required=True, help="Hypothesis file (plain text, one line per sentence).")
    parser.add_argument("--ref", nargs="+", required=True, help="Reference file(s), one or more.")
    parser.add_argument(
        "--scorers",
        nargs="+",
        metavar="scorer",
        choices=CORPUS_SCORERS,
        default=[],
        help=f"List of scorers - {CORPUS_SCORERS}. Default: {sorted(DEFAULT_SCORERS)}",
    )
    parser.add_argument("--out", default="scores.csv", help="Output scores CSV path. Default: ./scores.csv.")
    parser.add_argument(
        "--verse-scores",
        default=False,
        action="store_true",
        help="Write a per-verse *.scores.csv file.",
    )
    parser.add_argument("--verse-scores-out", help="Path for the per-verse CSV. Default: <hyp>.scores.csv.")
    args = parser.parse_args()

    scorers = set(s.lower() for s in args.scorers) if args.scorers else set(DEFAULT_SCORERS)

    hyp_path = Path(args.hyp)
    hyp_lines = _read_all_lines(hyp_path)
    ref_paths = [Path(p) for p in args.ref]
    ref_lines_list = [_read_all_lines(p) for p in ref_paths]
    lengths = {len(hyp_lines)} | {len(r) for r in ref_lines_list}
    if len(lengths) > 1:
        raise SystemExit(f"Input files have different line counts {sorted(lengths)}.")

    ref_names = {p.stem for p in ref_paths}
    if len(ref_names) != len(ref_paths):
        ref_names = {str(p) for p in ref_paths}

    scores = compute_corpus_scores(hyp_lines, ref_lines_list, scorers, DEFAULT_SACREBLEU_TOKENIZE)
    score = PairScore("ALL", "", "", scores, len(hyp_lines), ref_names)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as scores_file:
        score.writeHeader(scores_file)
        score.write(scores_file)

    if args.verse_scores:
        verse_path = (
            Path(args.verse_scores_out)
            if args.verse_scores_out
            else hyp_path.with_name(hyp_path.name + VERSE_SCORES_SUFFIX)
        )
        write_verse_scores(verse_path, hyp_lines, ref_lines_list, scorers, DEFAULT_SACREBLEU_TOKENIZE, ref_names)


if __name__ == "__main__":
    main()
