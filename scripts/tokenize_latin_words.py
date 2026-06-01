import argparse
from pathlib import Path

from machine.tokenization import LatinWordTokenizer


def tokenize_file(input_path: Path, output_path: Path) -> None:
    tokenizer = LatinWordTokenizer()

    with input_path.open("r", encoding="utf-8-sig") as input_file, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line in input_file:
            text = line.rstrip("\r\n")
            output_file.write(" ".join(tokenizer.tokenize(text)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a text file line-by-line using LatinWordTokenizer.")
    parser.add_argument("input", type=Path, help="Path to the input text file")
    parser.add_argument("output", type=Path, help="Path to the tokenized output text file")
    args = parser.parse_args()

    tokenize_file(args.input, args.output)


if __name__ == "__main__":
    main()
