# Adapted from https://github.com/valentinmace/noisy-text (Valentin Mace (valentin.mace@kedgebs.com)
#
# Add noise to your corpus

import argparse
from typing import Iterable, List, cast

from tqdm import tqdm

from .utils import DeleteRandomToken, NoiseMethod, RandomTokenPermutation, ReplaceRandomToken


def count_lines(file_path: str) -> int:
    return sum(1 for _ in open(file_path, "r", encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="The text file you want to add noise to")
    parser.add_argument(
        "--output", default=None, help="Optional, the name you want to give to your output, default=yourfilename.noisy"
    )
    parser.add_argument("--progress", action="store_true", help="Optional, show the progress")
    parser.add_argument(
        "--delete_probability",
        default=0.1,
        type=float,
        help="Optional, the probability to remove each token, default=0.1",
    )
    parser.add_argument(
        "--replace_probability",
        default=0.1,
        type=float,
        help="Optional, the probability to replace each token with a filler token, default=0.1",
    )
    parser.add_argument(
        "--permutation_range", default=3, type=int, help="Optional, Max range for token permutation, default=3"
    )
    parser.add_argument(
        "--filler_token", default="<blank>", help="Optional, token to use for replacement function, default=<blank>"
    )
    args = parser.parse_args()

    file_input: str = args.input
    file_output: str = file_input + ".noisy"
    if args.output:
        file_output = args.output

    lines_number = count_lines(file_input) if args.progress else None

    noise_methods: List[NoiseMethod] = [
        DeleteRandomToken(args.delete_probability),
        ReplaceRandomToken(args.replace_probability, filler_token=args.filler_token),
        RandomTokenPermutation(args.permutation_range),
    ]

    with open(file_input, "r", encoding="utf-8") as corpus, open(
        file_output, "w", encoding="utf-8", newline="\n"
    ) as output:
        # You can remove a noise function here, modify its parameters or add your own (writing it in noise_functions.py)
        for line in cast(Iterable[str], tqdm(corpus, total=lines_number)):
            tokens = line.split()
            for noise_method in noise_methods:
                tokens = noise_method(tokens)
            output.write(" ".join(tokens) + "\n")


if __name__ == "__main__":
    main()
