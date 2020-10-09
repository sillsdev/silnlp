import argparse
import glob
import os
from typing import List

from nltk.translate import Alignment

from nlp.alignment.config import get_aligner
from nlp.alignment.metrics import compute_aer, compute_f_score
from nlp.common.corpus import load_corpus
from nlp.common.utils import get_align_root_dir


def load_alignments(input_file: str) -> List[Alignment]:
    alignments: List[Alignment] = []
    for line in load_corpus(input_file):
        if line.startswith("#"):
            continue
        alignments.append(Alignment.fromstring(line))
    return alignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests generated alignments against gold standard alignments")
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    root_dir = get_align_root_dir(args.experiment)

    ref_file_path = os.path.join(root_dir, "alignments.gold.txt")
    references = load_alignments(ref_file_path)

    with open(os.path.join(root_dir, "scores.csv"), "w", encoding="utf-8") as scores_file:
        scores_file.write("Model,AER,F-Score,Precision,Recall\n")
        for alignments_path in glob.glob(os.path.join(root_dir, "alignments.*.txt")):
            if alignments_path == ref_file_path:
                continue
            file_name = os.path.basename(alignments_path)
            parts = file_name.split(".")
            id = parts[1]
            aligner = get_aligner(id, root_dir)

            alignments = load_alignments(alignments_path)

            aer = compute_aer(alignments, references)
            f_score, precision, recall = compute_f_score(alignments, references)

            print(aligner.name)
            print(f"- AER: {aer:.4f}")
            print(f"- F-Score: {f_score:.4f}")
            print(f"- Precision: {precision:.4f}")
            print(f"- Recall: {recall:.4f}")
            scores_file.write(f"{aligner.name},{aer:.4f},{f_score:.4f},{precision:.4f},{recall:.4f}\n")


if __name__ == "__main__":
    main()
