import argparse
from typing import List

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filters sentence pairs from a parallel corpus using the provided scores"
    )
    parser.add_argument("--input", type=str, help="The tab-delimited input corpus file")
    parser.add_argument("--scores", type=str, help="The scores file")
    parser.add_argument("--src-output", type=str, help="The output source corpus file")
    parser.add_argument("--trg-output", type=str, help="The output target corpus file")
    parser.add_argument("--threshold", type=float, required=False, help="The score threshold")
    parser.add_argument("--percentage", type=float, required=False, help="The percentage of the corpus to filter")
    args = parser.parse_args()

    scores: List[float] = []
    with open(args.scores, "r", encoding="utf-8") as scores_file:
        for line in scores_file:
            line = line.strip()
            scores.append(float(line))

    threshold: float = 0
    max_count = len(scores)
    if args.threshold is not None:
        threshold = args.threshold
    elif args.percentage is not None:
        threshold = np.quantile(scores, args.percentage)
        max_count = int(len(scores) * (1 - args.percentage))
    else:
        print("Threshold or percentage must be specified.")
        return

    count: int = 0
    with open(args.input, "r", encoding="utf-8") as input_file, open(
        args.src_output, "w", encoding="utf-8"
    ) as src_output_file, open(args.trg_output, "w", encoding="utf-8") as trg_output_file:
        for sentences, score in zip(input_file, scores):
            if score > threshold:
                src_sentence, trg_sentence = sentences.strip().split("\t")
                src_output_file.write(src_sentence + "\n")
                trg_output_file.write(trg_sentence + "\n")
                count += 1
                if count == max_count:
                    break


if __name__ == "__main__":
    main()
