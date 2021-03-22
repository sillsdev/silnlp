import argparse
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from ..alignment.utils import compute_alignment_scores


def filterTsvFile(
    tsv_path: str,
    scores_path: str,
    src_out_path: str,
    trg_out_path: str,
    threshold: float = None,
    percentage: float = None,
):
    # Load the scores
    scores: List[float] = []
    with open(scores_path, "r", encoding="utf-8") as scores_file:
        for line in scores_file:
            line = line.strip()
            scores.append(float(line))

    # Calculate the filtering threshold and maximum # of lines
    max_count = len(scores)
    if threshold is None:
        threshold = 0
    elif percentage is not None:
        threshold = np.quantile(scores, percentage)
        max_count = int(len(scores) * (1 - percentage))
    assert threshold is not None

    # Write the line pairs with scores exceeding the threshold
    count: int = 0
    with open(tsv_path, "r", encoding="utf-8") as input_file, open(
        src_out_path, "w", encoding="utf-8"
    ) as src_output_file, open(trg_out_path, "w", encoding="utf-8") as trg_output_file:
        for sentences, score in zip(input_file, scores):
            if score > threshold:
                src_sentence, trg_sentence = sentences.strip().split("\t")
                src_output_file.write(src_sentence + "\n")
                trg_output_file.write(trg_sentence + "\n")
                count += 1
                if count == max_count:
                    break


def check_sent_pair(src_sent: str, trg_sent: str) -> bool:
    if src_sent != "\n" and trg_sent != "\n":
        return True
    return False


def split_and_filter_tsv(input_path: Path, src_path: Path, trg_path: Path):
    with open(input_path, "r", encoding="utf-8") as input_file, open(
        src_path, "w", encoding="utf-8"
    ) as src_output_file, open(trg_path, "w", encoding="utf-8") as trg_output_file:
        for sentences in input_file:
            src_sent, trg_sent = sentences.strip().split("\t")
            if check_sent_pair(src_sent, trg_sent):
                src_output_file.write(src_sent + "\n")
                trg_output_file.write(trg_sent + "\n")


def filter_src_trg_files(src_in_path: Path, trg_in_path: Path, src_out_path: Path, trg_out_path: Path):
    with open(src_in_path, "r", encoding="utf-8") as src_file, open(
        trg_in_path, "r", encoding="utf-8"
    ) as trg_file, open(src_out_path, "w", encoding="utf-8") as tmp_src_file, open(
        trg_out_path, "w", encoding="utf-8"
    ) as tmp_trg_file:
        for src_sent, trg_sent in zip(src_file, trg_file):
            if check_sent_pair(src_sent, trg_sent):
                tmp_src_file.write(src_sent)
                tmp_trg_file.write(trg_sent)


def write_scores_file(scores_out_path: Path, scores: List[float], src_path: Path, trg_path: Path):
    with open(src_path, "r", encoding="utf-8") as src_in, open(trg_path, "r", encoding="utf-8") as trg_in, open(
        scores_out_path, "w", encoding="utf-8"
    ) as scores_out:
        for src_sent, trg_sent, score in zip(src_in, trg_in, scores):
            scores_out.write(f"{src_sent.strip()}\t{trg_sent.strip()}\t{score:.3f}\n")


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
    parser.add_argument("--scores-output", type=str, required=False, help="The scores output file")
    parser.add_argument("--threshold", type=float, required=False, help="The score threshold")
    parser.add_argument("--percentage", type=float, required=False, help="The percentage of the corpus to filter")
    args = parser.parse_args()

    if args.input is None and (args.src_input is None or args.trg_input is None):
        print("--input or (--src-input and --target-input) must be specified.")
        return

    if args.threshold is None and args.percentage is None:
        print("Threshold or percentage must be specified.")
        return

    if args.input is not None and args.scores is not None:
        filterTsvFile(args.input, args.scores, args.src_output, args.trg_output, args.threshold, args.percentage)
        return

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        src_path = temp_dir / "tmp_src.txt"
        trg_path = temp_dir / "tmp_trg.txt"

        # Generate temporary src/trg files with basic filtering (empty sentences; length checks; etc)
        if args.input is not None:
            split_and_filter_tsv(args.input, src_path, trg_path)
        elif args.src_input is not None and args.trg_input is not None:
            filter_src_trg_files(args.src_input, args.trg_input, src_path, trg_path)

        # Generate alignment scores to use for alignment filtering
        scores = compute_alignment_scores(src_path, trg_path)

        # Assign the alignment filtering settings
        threshold: float = 0
        max_count = len(scores)
        if args.threshold is not None:
            threshold = args.threshold
        elif args.percentage is not None:
            threshold = np.quantile(scores, args.percentage)
            max_count = int(len(scores) * (1 - args.percentage))

        if args.scores_output is not None:
            write_scores_file(args.scores_output, scores, src_path, trg_path)

        # Write the filtered results
        count: int = 0
        with open(src_path, "r", encoding="utf-8") as src_in, open(trg_path, "r", encoding="utf-8") as trg_in, open(
            args.src_output, "w", encoding="utf-8"
        ) as src_out, open(args.trg_output, "w", encoding="utf-8") as trg_out:
            for src_sent, trg_sent, score in zip(src_in, trg_in, scores):
                if score > threshold:
                    src_out.write(src_sent)
                    trg_out.write(trg_sent)
                    count += 1
                    if count == max_count:
                        break


if __name__ == "__main__":
    main()
