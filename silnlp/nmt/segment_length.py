# create empty list to hold series of segment lengths
# with open file
#   for each line in file
#     add length of segment in tokens to the list
# plot the histogram of the data, choose bin numbers by experiment
# log number of lengths over 200


import argparse
import logging

import matplotlib.pyplot as plt

from ..common.environment import SIL_NLP_ENV
from ..common.utils import get_mt_exp_dir

logging.basicConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a histogram of segment lengths in tokens")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("filename", help="Tokenized file in experiment folder")
    args = parser.parse_args()

    exp_name = args.experiment
    SIL_NLP_ENV.copy_experiment_from_bucket(exp_name)

    exp_dir = get_mt_exp_dir(exp_name)
    data = []
    with open(exp_dir / args.filename, "r+", encoding="utf-8") as f:
        for line in f:
            data.append(len(line.split()))

    print(f"Num seg lengths >= 200: {sum(seg_length >= 200 for seg_length in data)}")
    print(f"Max seg length: {max(data)}")
    print(f"Avg seg length: {sum(data)/len(data)}")

    plt.hist(data, bins=20, color="blue", alpha=0.7)
    plt.xlabel("Segment Length (tokens)")
    plt.ylabel("Number of Segments")
    plt.title("Distribution of Segment Lengths")
    plt.savefig(exp_dir / "histogram_seg_length.png")
    SIL_NLP_ENV.copy_experiment_to_bucket(exp_name)


if __name__ == "__main__":
    main()
