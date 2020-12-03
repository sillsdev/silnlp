import argparse
from typing import List

import numpy as np

from .corpus import compute_alignment_scores

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filters sentence pairs from a parallel corpus using the provided scores"
    )
    parser.add_argument("--input", type=str, required=False, help="The tab-delimited input corpus file")
    parser.add_argument("--src-input", type=str, required=False, help="The input source corpus file")
    parser.add_argument("--trg-input", type=str, required=False, help="The input source corpus file")
    parser.add_argument("--scores", type=str, required=False, help="The scores file")
    parser.add_argument("--src-output", type=str, help="The output source corpus file")
    parser.add_argument("--trg-output", type=str, help="The output target corpus file")
    parser.add_argument("--threshold", type=float, required=False, help="The score threshold")
    parser.add_argument("--percentage", type=float, required=False, help="The percentage of the corpus to filter")
    args = parser.parse_args()

    scores: List[float] = []
    if args.scores is not None:
        # Read the scores file
        with open(args.scores, "r", encoding="utf-8") as scores_file:
            for line in scores_file:
                line = line.strip()
                scores.append(float(line))
    else:
        if args.input is not None:
            # Calculate the scores for a tab-delimited input corpus
            scores = compute_alignment_scores(args.input, None)
        elif args.src_input is not None and args.trg_input is not None:
            # Calculate the scores from separate (parallel) src/trg input files
            scores = compute_alignment_scores(args.src_input, args.trg_input)
        else:
            print("--input or (--src-input and --target-input) must be specified.")
            return

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
    if args.input is not None:
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
    else:
        with open(args.src_input, "r", encoding="utf-8") as src_in,\
             open(args.trg_input, "r", encoding="utf-8") as trg_in,\
             open(args.src_output, "w", encoding="utf-8") as src_out,\
             open(args.trg_output, "w", encoding="utf-8") as trg_out:
            for src_sent, trg_sent, score in zip(src_in, trg_in, scores):
                if score > threshold:
                    src_out.write(src_sent + "\n")
                    trg_out.write(trg_sent + "\n")
                    count += 1
                    if count == max_count:
                        break

if __name__ == "__main__":
    main()
