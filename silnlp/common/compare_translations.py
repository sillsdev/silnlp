import argparse
from pathlib import Path
from typing import Dict, List, Set

import sacrebleu
from machine.corpora import ParatextTextCorpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare translations")
    parser.add_argument(
        "--projects", nargs=2, type=Path, required=True, help="Paths to the Paratext project directories"
    )
    parser.add_argument(
        "--output-file", type=Path, required=False, help="Path to the file to save the comparison scores"
    )
    parser.add_argument(
        "--scorers",
        nargs="*",
        metavar="scorer",
        default={"bleu", "chrf3", "chrf3+", "chrf3++", "spbleu", "ter"},
        help="Set of scorers",
    )
    parser.add_argument(
        "--score-empty", type=bool, required=False, help="If true, also calculate BLEU score on segment pairs where at least one segment is empty", default=False
    )

    args = parser.parse_args()
    scores = compare_translations(args.projects[0], args.projects[1], args.scorers, args.score_empty)

    print(f"{args.projects[0]},{args.projects[1]}")
    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            f.write(f"{args.projects[0]},{args.projects[1]}\n")
            for key, value in scores.items():
                f.write(f"{key},{value}\n")
                print(f"{key},{value}")
    else:
        for key, value in scores.items():
            print(f"{key},{value}")


def compare_translations(project1: Path, project2: Path, scorers: Set[str], score_empty:bool=False) -> Dict[str, float]:
    corpus_a = ParatextTextCorpus(project1)
    corpus_b = ParatextTextCorpus(project2)
    parallel_corpus = corpus_a.align_rows(corpus_b)
    a_lines = []
    b_lines = []
    with parallel_corpus.get_rows() as rows:
        for row in rows:
            if not score_empty and row.is_empty:
                continue
            a_lines.append(row.source_text)
            b_lines.append(row.target_text)
    return score_pair(a_lines, [b_lines], scorers)


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

    if "ter" in scorers:
        ter_score = sacrebleu.corpus_ter(pair_sys, pair_refs)
        if ter_score.score >= 0:
            scores["TER"] = ter_score.score

    return scores


if __name__ == "__main__":
    main()
