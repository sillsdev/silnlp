"""Score translation output produced outside silnlp,
using the same metrics and parameters as silnlp's test stage.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Set

import sacrebleu

from .scoring_metrics import (
    CORPUS_SCORERS,
    DEFAULT_SACREBLEU_TOKENIZE,
    SENTENCE_SCORERS,
    PairScore,
    compute_corpus_scores,
    iter_verse_scores,
)

DEFAULT_SCORERS = {"bleu", "chrf3", "chrf3+", "chrf3++"}
VERSE_SCORES_SUFFIX = ".scores.tsv"
EXPECTED_SACREBLEU_VERSION = "2.6.0"


def check_version() -> None:
    if sacrebleu.__version__ != EXPECTED_SACREBLEU_VERSION:
        raise SystemExit(
            f"Version mismatch - scores may not match silnlp: sacrebleu {sacrebleu.__version__} "
            f"(silnlp pins {EXPECTED_SACREBLEU_VERSION})"
        )


def write_verse_scores(
    path: Path,
    pair_sys: List[str],
    pair_refs: List[List[str]],
    scorers: Set[str],
    sacrebleu_tokenize: str,
) -> None:
    scorers = scorers.intersection(SENTENCE_SCORERS)

    with open(path, "w", encoding="utf-8", newline="") as scores_file:
        writer = csv.writer(scores_file, delimiter="\t")
        header = ["Verse"]
        if "bleu" in scorers:
            header += [
                "BLEU",
                "BLEU_1gram_prec",
                "BLEU_2gram_prec",
                "BLEU_3gram_prec",
                "BLEU_4gram_prec",
                "BLEU_brevity_penalty",
            ]
        metric_columns = [s for s in ["chrF3", "chrF3+", "chrF3++", "spBLEU", "TER"] if s.lower() in scorers]
        header += metric_columns
        header.append("Prediction")
        for _ in pair_refs:
            header.append("Reference")
        writer.writerow(header)

        for index, pred, sentences, bleu_verse_score, other_verse_scores in iter_verse_scores(
            pair_sys, pair_refs, scorers, sacrebleu_tokenize
        ):
            row: List[str] = [f"{index + 1}"]
            if "bleu" in scorers:
                assert bleu_verse_score is not None
                row += [
                    f"{bleu_verse_score.score:.2f}",
                    f"{bleu_verse_score.precisions[0]:.2f}",
                    f"{bleu_verse_score.precisions[1]:.2f}",
                    f"{bleu_verse_score.precisions[2]:.2f}",
                    f"{bleu_verse_score.precisions[3]:.2f}",
                    f"{bleu_verse_score.bp:.3f}",
                ]
            if list(other_verse_scores.keys()) != metric_columns:
                raise ValueError(
                    f"The scores of verse {index + 1}, {list(other_verse_scores.keys())}, do not match the "
                    f"columns of the header, {metric_columns}."
                )
            for val in other_verse_scores.values():
                row.append(f"{val:.2f}")
            row.append(pred.rstrip("\n"))
            for sentence in sentences:
                row.append(sentence.rstrip("\n"))
            writer.writerow(row)


def _read_all_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def main() -> None:
    supported_tokenizers = sorted(sacrebleu.BLEU.TOKENIZERS)

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
    parser.add_argument(
        "--sacrebleu-tokenize",
        metavar="tokenizer",
        choices=supported_tokenizers,
        default=DEFAULT_SACREBLEU_TOKENIZE,
        help=(f"Available tokenizers - {supported_tokenizers}. Default: {DEFAULT_SACREBLEU_TOKENIZE}."),
    )
    parser.add_argument("--out", default="scores.csv", help="Output scores CSV path. Default: ./scores.csv.")
    parser.add_argument(
        "--verse-scores",
        default=False,
        action="store_true",
        help="Write a per-verse *.scores.tsv file.",
    )
    parser.add_argument("--verse-scores-out", help="Path for the per-verse TSV. Default: <out>.scores.tsv.")
    args = parser.parse_args()

    check_version()

    scorers = set(s.lower() for s in args.scorers) if args.scorers else set(DEFAULT_SCORERS)

    hyp_lines = _read_all_lines(Path(args.hyp))
    ref_paths = [Path(p) for p in args.ref]
    ref_lines_list = [_read_all_lines(p) for p in ref_paths]
    lengths = {len(hyp_lines)} | {len(r) for r in ref_lines_list}
    if len(lengths) > 1:
        raise SystemExit(f"Input files have different line counts {sorted(lengths)}.")

    ref_names = {p.stem for p in ref_paths}
    if len(ref_names) != len(ref_paths):
        ref_names = {str(p) for p in ref_paths}

    bleu_score, other_scores = compute_corpus_scores(hyp_lines, ref_lines_list, scorers, args.sacrebleu_tokenize)
    score = PairScore("ALL", "", "", bleu_score, len(hyp_lines), ref_names, other_scores)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as scores_file:
        score.writeHeader(scores_file)
        score.write(scores_file)

    if args.verse_scores:
        verse_path = (
            Path(args.verse_scores_out)
            if args.verse_scores_out
            else out_path.with_name(out_path.name + VERSE_SCORES_SUFFIX)
        )
        write_verse_scores(verse_path, hyp_lines, ref_lines_list, scorers, args.sacrebleu_tokenize)


if __name__ == "__main__":
    main()
