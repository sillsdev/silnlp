import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import sacrebleu

from silnlp.common.metrics import compute_wer_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare translations")
    parser.add_argument("--dir-path", type=Path, required=True, help="Path to the directory")
    parser.add_argument("--file-path-a", type=Path, required=True, help="Path to the first file from the directory")
    parser.add_argument("--file-path-b", type=Path, required=True, help="Path to the second file from the directory")
    parser.add_argument(
        "--scorers",
        nargs="*",
        metavar="scorer",
        default={"bleu", "chrf3", "chrf3+", "chrf3++", "spbleu", "wer", "ter"},
        help="Set of scorers",
    )
    args = parser.parse_args()

    file_a = args.dir_path.joinpath(args.file_path_a)
    file_b = args.dir_path.joinpath(args.file_path_b)
    scores = compare_translations(file_a, file_b, args.scorers)

    with open(f"{args.dir_path}/comparison_scores.txt", "w") as f:
        f.write(f"Comparison of Translations in Files: {args.file_path_a} and {args.file_path_b}\n")
        for key, value in scores.items():
            f.write(f"{key}: {value}\n")


def compare_translations(file_a: Path, file_b: Path, scorers: Set[str]) -> Dict[str, float]:
    try:
        with open(file_a, "r") as f:
            a_lines = f.readlines()
            # Remove usfm markers
            if file_a.name.lower().endswith(".usfm") or file_a.name.lower().endswith(".sfm"):
                print("Removing USFM markers")
                a_lines = a_lines[8:]
                a_lines = [re.sub(r"^\\\S+(\s[0-9]+)?", "", line) for line in a_lines]
        with open(file_b, "r") as f:
            b_lines = f.readlines()
            # Remove usfm markers
            if file_b.name.lower().endswith(".usfm") or file_b.name.lower().endswith(".sfm"):
                print("Removing USFM markers")
                b_lines = b_lines[8:]
                b_lines = [re.sub(r"^\\\S+(\s[0-9]+)?", "", line) for line in b_lines]

        return score_pair(a_lines, [b_lines], scorers)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return


def score_pair(pair_sys: List[str], pair_refs: List[List[str]], scorers: Set[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    if "bleu" in scorers:
        bleu_score = sacrebleu.corpus_bleu(
            pair_sys,
            pair_refs,
            lowercase=True,
        )
        scores["BLEU"] = bleu_score.score

    if "chrf3" in scorers:
        chrf3_score = sacrebleu.corpus_chrf(pair_sys, pair_refs, char_order=6, beta=3, remove_whitespace=True)
        scores["chrF3"] = chrf3_score.score

    if "chrf3+" in scorers:
        chrfp_score = sacrebleu.corpus_chrf(
            pair_sys, pair_refs, char_order=6, beta=3, word_order=1, remove_whitespace=True, eps_smoothing=True
        )
        scores["chrF3+"] = chrfp_score.score

    if "chrf3++" in scorers:
        chrfpp_score = sacrebleu.corpus_chrf(
            pair_sys, pair_refs, char_order=6, beta=3, word_order=2, remove_whitespace=True, eps_smoothing=True
        )
        scores["chrF3++"] = chrfpp_score.score

    if "spbleu" in scorers:
        spbleu_score = sacrebleu.corpus_bleu(
            pair_sys,
            pair_refs,
            lowercase=True,
            tokenize="flores200",
        )
        scores["spBLEU"] = spbleu_score.score

    if "wer" in scorers:
        wer_score = compute_wer_score(pair_sys, pair_refs)
        if wer_score >= 0:
            scores["WER"] = wer_score

    if "ter" in scorers:
        ter_score = sacrebleu.corpus_ter(pair_sys, pair_refs)
        if ter_score.score >= 0:
            scores["TER"] = ter_score.score

    return scores


if __name__ == "__main__":
    main()
