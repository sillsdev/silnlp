# Adapted from https://github.com/valentinmace/noisy-text (Valentin Mace (valentin.mace@kedgebs.com)
#
# Add noise to your corpus

import argparse
from tqdm import tqdm
import random

#### Utility functions

def random_bool(probability=0.5):
    """Returns True with given probability
    Args:
        probability: probability to return True
    """
    assert (0 <= probability <= 1), "probability needs to be >= 0 and <= 1"
    return random.random() < probability


def count_lines(filename):
    """Returns the number of lines in the given file
    Args:
        filename: (string) path to the file
    """
    return sum(1 for line in open(filename, "r", encoding="utf-8"))


### Functions for adding noise to text

def delete_random_token(line, probability):
    """Delete random tokens in a given String with given probability
    Args:
        line: a String
        probability: probability to delete each token
    """
    line_split = line.split()
    ret = [token for token in line_split if not random_bool(probability)]
    return " ".join(ret)


def replace_random_token(line, probability, filler_token="[[BLANK]]"):
    """Replace random tokens in a String by a filler token with given probability
    Args:
        line: a String
        probability: probability to replace each token
        filler_token: token replacing chosen tokens
    """
    line_split = line.split()
    for i in range(len(line_split)):
        if random_bool(probability):
            line_split[i] = filler_token
    return " ".join(line_split)


def random_token_permutation(line, _range):
    """Random permutation over the tokens of a String, restricted to a range, drawn from the uniform distribution
    Args:
        line: a String
        _range: Max range for token permutation
    """
    line_split = line.split()
    new_indices = [i+random.uniform(0, _range+1) for i in range(len(line_split))]
    res = [x for _, x in sorted(zip(new_indices, line_split), key=lambda pair: pair[0])]
    return " ".join(res)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        help="The text file you want to add noise to")
    parser.add_argument('--output', default=None,
                        help="Optional, the name you want to give to your output, default=yourfilename.noisy")
    parser.add_argument('--progress', action='store_true',
                        help="Optional, show the progress")
    parser.add_argument('--delete_probability', default=0.1, type=float,
                        help="Optional, the probability to remove each token, default=0.1")
    parser.add_argument('--replace_probability', default=0.1, type=float,
                        help="Optional, the probability to replace each token with a filler token, default=0.1")
    parser.add_argument('--permutation_range', default=3, type=int,
                        help="Optional, Max range for token permutation, default=3")
    parser.add_argument('--filler_token', default='[[BLANK]]',
                        help="Optional, token to use for replacement function, default=[[BLANK]]")
    args = parser.parse_args()

    file_input = args.input
    file_output = file_input + ".noisy"
    if args.output:
        file_output = args.output

    lines_number = count_lines(file_input) if args.progress else None

    with open(file_input, 'r', encoding='utf-8') as corpus, open(file_output, 'w', encoding='utf-8') as output:
        # You can remove a noise function here, modify its parameters or add your own (writing it in noise_functions.py)
        for line in tqdm(corpus, total=lines_number):
            line = delete_random_token(line, probability=args.delete_probability)
            line = replace_random_token(line, probability=args.replace_probability, filler_token=args.filler_token)
            line = random_token_permutation(line, _range=args.permutation_range)
            output.write(line + '\n')


if __name__ == "__main__":
    main()
