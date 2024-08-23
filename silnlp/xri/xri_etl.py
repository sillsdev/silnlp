"""
Transforms tsv files in the XRI format to extract files suitable for training SIL machine translation models.

Each execution of this script creates a unique directory within the output directory containing the output files.

Run with --help for more details.

See https://github.com/sillsdev/silnlp/issues/472 for original context.
"""
import argparse
import csv

from dataclasses import dataclass
from enum import Enum
from typing import List


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass(frozen=True)
class SentencePair:
    id: int
    source: str
    target: str
    split: Split


@dataclass(frozen=True)
class CliInput:
    input_file_path: str
    source_iso: str
    target_iso: str
    dataset_descriptor: str


def load_sentence_pairs(input_file_path: str) -> List[SentencePair]:
    with open(input_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        rows = list(reader)

    def clean(s: str) -> str:
        return s.lower().strip()

    header_row = [clean(column) for column in rows[0]]
    data_rows = rows[1:]

    def get_column_index(column_name: str) -> int:
        index = header_row.index(clean(column_name))
        if index >= 0:
            return index
        else:
            raise Exception(f"Unable to find expected column '{column_name}' in input file")

    id_column_index = get_column_index("id")
    source_column_index = get_column_index("source")
    target_column_index = get_column_index("target")
    split_column_index = get_column_index("split")

    return [
        SentencePair(
            id=int(row[id_column_index]),
            source=row[source_column_index],
            target=row[target_column_index],
            split=Split[row[split_column_index]],
        )
        for row in data_rows
    ]


def create_extract_files(sentence_pairs: List[SentencePair]) -> None:
    return


def run(cli_input: CliInput) -> None:
    sentence_pairs = load_sentence_pairs(cli_input.input_file_path)
    create_extract_files(sentence_pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transforms XRI tsv files into extract files")

    parser.add_argument("input_file", help="Path to the input tsv file with UTF-8 encoding", type=str)
    parser.add_argument("source_iso", help="The ISO 693-3 code for the source/LWC language", type=str)
    parser.add_argument("target_iso", help="The ISO 693-3 code for the target/vernacular language", type=str)
    parser.add_argument("dataset", help="A descriptor of the dataset to be used in the output filename", type=str)
    args = parser.parse_args()

    cli_input = CliInput(
        input_file_path=args.input_file,
        source_iso=args.source_iso,
        target_iso=args.target_iso,
        dataset_descriptor=args.dataset,
    )
    run(cli_input)


if __name__ == "__main__":
    main()
