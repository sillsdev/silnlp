"""
Transforms tsv files in the XRI format to extract files suitable for training SIL machine translation models.

Example:

    $ python -m silnlp.common.extract_xri   data.tsv   swa     ngq     XRI-2024-08-14
                                               ^        ^       ^           ^
                                             input   source    target     dataset
                                             file    iso code  iso code   descriptor

Each execution puts the extract files into a unique directory in `out` with name generated from script inputs and the current time:

    out
    └── swa-ngq-XRI-2024-08-14-20240822-180428
        ├── swa-XRI-2024-08-14.all.txt     ┓ complete extract files (no filtering)
        ├── ngq-XRI-2024-08-14.all.txt     ┛
        ├── swa-XRI-2024-08-14.train.txt   ┓ extract files filtered to training data
        ├── ngq-XRI-2024-08-14.train.txt   ┛
        ├── swa-XRI-2024-08-14.dev.txt     ┓ extract files filtered to dev/validation data
        ├── ngq-XRI-2024-08-14.dev.txt     ┛
        ├── swa-XRI-2024-08-14.test.txt    ┓ extract files filtered to test data
        └── ngq-XRI-2024-08-14.test.txt    ┛
    (Note that a subsequent PR is going to rework the output location to follow SIL team conventions)

Run with --help for more details.

See https://github.com/sillsdev/silnlp/issues/472 for original context.
"""

import argparse
import csv
import os
import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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


def write_output_file(filename: str, sentences: List[str]) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(f"{sentence}{os.linesep}")


def create_extract_files(cli_input: CliInput, sentence_pairs: List[SentencePair]) -> None:
    unique_output_dir = (
        f"{cli_input.source_iso}-{cli_input.target_iso}-{cli_input.dataset_descriptor}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    Path(os.path.join("out", unique_output_dir)).mkdir(parents=True, exist_ok=True)

    def create_source_target_files(sub_sentence_pairs: List[SentencePair], suffix: str) -> None:
        def build_output_path(iso: str) -> str:
            return os.path.join("out", unique_output_dir, f"{iso}-{cli_input.dataset_descriptor}.{suffix}.txt")

        source_filename = build_output_path(iso=cli_input.source_iso)
        source_sentences = [sentence.source for sentence in sub_sentence_pairs]
        write_output_file(source_filename, source_sentences)

        target_filename = build_output_path(iso=cli_input.target_iso)
        target_sentences = [sentence.target for sentence in sub_sentence_pairs]
        write_output_file(target_filename, target_sentences)

    # *.all.txt
    create_source_target_files(sentence_pairs, "all")

    # *.train.txt
    train_sentences = [sentence_pair for sentence_pair in sentence_pairs if sentence_pair.split == Split.train]
    create_source_target_files(train_sentences, "train")

    # *.val.txt
    # NOTE that input data uses "dev" but it's converted to "val" here to make working with downstream SIL tools easier
    dev_sentences = [sentence_pair for sentence_pair in sentence_pairs if sentence_pair.split == Split.dev]
    create_source_target_files(dev_sentences, "val")

    # *.test.txt
    test_sentences = [sentence_pair for sentence_pair in sentence_pairs if sentence_pair.split == Split.test]
    create_source_target_files(test_sentences, "test")


def run(cli_input: CliInput) -> None:
    sentence_pairs = load_sentence_pairs(cli_input.input_file_path)
    create_extract_files(cli_input, sentence_pairs)


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
