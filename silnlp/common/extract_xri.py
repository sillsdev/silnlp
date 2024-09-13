"""
Transforms tsv files in the XRI format to extract files suitable for training SIL machine translation models.

Example:

    $ python -m silnlp.common.extract_xri   data.tsv   swa     ngq     XRI-2024-08-14
                                               ^        ^       ^           ^
                                             input   source    target     dataset
                                             file    iso code  iso code   descriptor

The output directory can be explicitly set using the `-output` arg, e.g.

    $ python -m silnlp.common.extract_xri data.tsv swa ngq XRI-2024-08-14 -output /tmp/test

Note that the script will write over any existing extract files in that directory.

If the output directory isn't set, a unique directory is created within the SIL_NLP_ENV.mt_corpora_dir (see common/environment.py).
The unique directory is constructed using the input arguments and current date time,
e.g. `swa-ngq-XRI-2024-08-14-20240822-180428` in the example above.

Files of the form `*.all.txt` and `*.(train/dev/test).txt` are created in the output directory for source and target languages.
The prefix of the filename is built from the cli inputs. For the example above these files would be created:

        ├── swa-XRI-2024-08-14.all.txt     ┓ complete extract files (no filtering)
        ├── ngq-XRI-2024-08-14.all.txt     ┛
        ├── swa-XRI-2024-08-14.train.txt   ┓ extract files filtered to training data
        ├── ngq-XRI-2024-08-14.train.txt   ┛
        ├── swa-XRI-2024-08-14.val.txt     ┓ extract files filtered to dev/validation data
        ├── ngq-XRI-2024-08-14.val.txt     ┛
        ├── swa-XRI-2024-08-14.test.txt    ┓ extract files filtered to test data
        └── ngq-XRI-2024-08-14.test.txt    ┛

Run with --help for more details.

By default the script uses the logging configuration inherited from the parent packages (which should log at INFO level).
There is detailed DEBUG level logging that can assist with troubleshooting.
You can enable DEBUG level logging by passing `-log_level DEBUG`.
Other accepted values are "INFO", "WARNING/WARN", "ERROR" and "CRITICAL".

See https://github.com/sillsdev/silnlp/issues/472 for original context.
"""

import argparse
import csv
import logging
import os
import time

from dataclasses import dataclass
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger(__package__ + ".extract_xri")
repair_logger = logging.getLogger(logger.name + ".repair")
clean_logger = logging.getLogger(logger.name + ".clean")


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
    transformation_applied: bool


@dataclass(frozen=True)
class ColumnSchema:
    """
    Represents the mapping of column indexes to data in the tsv.
    This is calculated on the fly off the indices of the column names in the header row.
    Most of the time it will 0, 1, 2, 3 respectively for the fields below.
    """

    id_column_index: int
    source_column_index: int
    target_column_index: int
    split_column_index: int


@dataclass(frozen=True)
class CliInput:
    input_file_path: str
    source_iso: str
    target_iso: str
    dataset_descriptor: str
    output: Optional[str]
    log_level: Optional[str]


def load_sentence_pairs(input_file_path: str) -> List[SentencePair]:
    logger.info("Loading sentence pairs")
    logger.debug("Opening file")
    with open(input_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        rows = list(reader)

    def clean(s: str) -> str:
        return s.lower().strip()

    logger.debug("Determining column indexes")
    header_row = [clean(column) for column in rows[0]]
    data_rows = rows[1:]

    def get_column_index(column_name: str) -> int:
        index = header_row.index(clean(column_name))
        if index >= 0:
            logger.debug(f"Column {column_name} found at index {index}")
            return index
        else:
            raise Exception(f"Unable to find expected column '{column_name}' in input file")

    column_schema = ColumnSchema(
        id_column_index=get_column_index("id"),
        source_column_index=get_column_index("source"),
        target_column_index=get_column_index("target"),
        split_column_index=get_column_index("split"),
    )

    logger.debug(
        f"Column indexes: id={column_schema.id_column_index} source={column_schema.source_column_index} "
        + f"target={column_schema.target_column_index} split={column_schema.split_column_index}"
    )

    logger.debug("Checking all rows contain 4 cells")
    all_rows_contain_four_cells = all(len(row) == 4 for row in data_rows)
    if not all_rows_contain_four_cells:
        # Potential causes:
        # - missing entries
        # - newlines embedded in source or target sentences that are splitting them
        # - trailing tab characters added causing extra rows
        logger.warning("Not all rows contain 4 cells")

    repaired = repair_if_necessary(column_schema.id_column_index, data_rows)

    return filter_and_clean(repaired, column_schema)


def try_extract_id(row: List[str], id_column_index: int, log: Logger) -> Optional[int]:
    """Tries to find a numerical id in the row passed at the expected position"""
    if id_column_index >= len(row):
        repair_logger.debug("Short row")
        return None
    else:
        id_str = row[id_column_index]
        if id_str.isdigit():
            return int(id_str)
        else:
            log.debug(f"Can't parse id cell: '{id_str}' - assuming row is broken")
            return None


def repair_if_necessary(id_column_index: int, rows: List[List[str]]) -> List[List[str]]:
    """
    Searches for rows that have been broken over 2 lines, and repairs them into a single line.
    This can happen sometimes due to newlines being inserted into the source or target sentences.
    """
    repair_logger.info("Repair starting")
    # If newlines were in the original tsv, then the row structure would be split up, e.g.
    #     [ID0, source0, target0, split0]
    #     [ID1, source1, tar]        | target1 broken over 2 lines
    #     [get1, split1]             |
    #     [ID2, source2, target2, split2]
    #     [ID3, so        |
    #     [urc]           | source3 broken over 3 lines
    #     [e3, targ]      |      | target3 broken over 2 lines
    #     [et3, split3]          |
    # It would repair to:
    #     [ID0, source0, target0, split0]
    #     [ID1, source1, target1, split1]
    #     [ID2, source2, target2, split2]
    #     [ID3, source3, target3, split3]
    #
    # The mechanism for detecting splits is to look for lines that don't have a numerical id in the position expected.
    # It's assumed that those lines should be joined up onto the lines prior.
    # NOTE: If a source/target string happened to end with a newline followed by a number,
    # then the number would split to the next line and be mistaken for an id.
    # This is very unlikely and solving it increases complexity so we don't repair that case.

    repaired: List[List[str]] = []
    # Represents the accumulated row data that is gradually populated when a sentence pair is split across many lines
    current_row: List[str] = []
    for row_index, next_row in enumerate(rows):
        # Ignore empty lines
        if not next_row:
            repair_logger.debug(f"Empty line detected at row index={row_index}")
            continue
        id_opt = try_extract_id(next_row, id_column_index, repair_logger)
        repair_logger.debug(f"Examining row index={row_index} id={id_opt}: {next_row}")
        if id_opt is not None:
            # We are starting a new row
            # Commit the previously accumulated row (except in the special case of the first row)
            if len(current_row) > 0:
                repair_logger.debug(f"New row starting at index={row_index}, committing previous row {current_row}")
                repaired.append(current_row)
            current_row = next_row
        else:
            # Merge the current row with the next row
            # The last cell of the previous row combines with the first cell of this row
            repair_logger.debug(f"Merging row {row_index} with data from previous row(s)")
            last_cell = current_row[-1]
            first_cell = next_row[0]
            current_row[-1] = last_cell + first_cell
            current_row.extend(next_row[1:])

    # The very last row needs to be committed as it doesn't have a following row
    repaired.append(current_row)

    repair_logger.info("Repair complete")
    return repaired


def filter_and_clean(
    rows: List[List[str]],
    column_schema: ColumnSchema,
) -> List[SentencePair]:
    """
    Applies basic checking and cleaning to the data.
    Rows of data that can be processed are cleaned and transformed to a structured SentencePair.
    Rows of data that can't be meaningfully processed are excluded.
    """
    sentence_pairs: List[SentencePair] = []

    clean_logger.info("Starting filtering and cleaning stage")
    required_columns = (
        max(
            [
                column_schema.id_column_index,
                column_schema.source_column_index,
                column_schema.target_column_index,
                column_schema.split_column_index,
            ]
        )
        + 1
    )
    clean_logger.debug(f"Required number of columns for each stage is {required_columns}")

    for row_index, row in enumerate(rows):
        transformation_applied = False
        clean_logger.debug(f"Processing row index={row_index}: {row}")

        if len(row) < required_columns:
            clean_logger.warning(
                f"Row found with only {len(row)} cells, at least {required_columns} expected. Ignoring: {row}"
            )
            continue

        id = try_extract_id(row, column_schema.id_column_index, clean_logger)

        if id is None:
            # This case should be virtually impossible based on how the repair logic works
            clean_logger.warning(
                f"Unable to identify id in row - potentially badly formatted or a bug in the repair logic. Ignoring: {row}"
            )
            continue

        def trim(sentence: str, description: str) -> str:
            trimmed = sentence.strip()
            if trimmed != sentence:
                clean_logger.debug(
                    f"Boundary whitespace trimmed off '{description}' field. "
                    + f"Number of trimmed chars: {len(sentence) - len(trimmed)}"
                )
                nonlocal transformation_applied
                transformation_applied = True
            return trimmed

        source = trim(row[column_schema.source_column_index], "source")
        target = trim(row[column_schema.target_column_index], "target")

        if target == "!":
            clean_logger.debug("Target sentence is '!' indicating it is not translated. Ignoring.")
            continue

        split_text = trim(row[column_schema.split_column_index], "split").lower()
        if split_text not in ["train", "dev", "test"]:
            clean_logger.warning(
                f"Split value '{split_text}' is not a recognized value. Keeping sentence but assigning to training data"
            )
            transformation_applied = True
            split = Split.train
        else:
            split = Split[split_text]

        clean_logger.debug(
            f"Successfully parsed and cleaned row index={row_index} id={id}. "
            + f"Transformations applied? {transformation_applied}"
        )

        sentence_pairs.append(
            SentencePair(
                id=id,
                source=source,
                target=target,
                split=split,
                transformation_applied=transformation_applied,
            )
        )

    num_modified = len([sentence_pair for sentence_pair in sentence_pairs if sentence_pair.transformation_applied])
    clean_logger.info(
        "Finished filtering and cleaning stage. "
        + f"{len(rows)} rows ingested. "
        + f"{len(sentence_pairs)} survived. "
        + f"{len(rows) - len(sentence_pairs)} removed. "
        + f"{num_modified} survivors were transformed in some way."
    )
    return sentence_pairs


def write_output_file(filepath: Path, sentences: List[str]) -> None:
    logger.debug(f"Writing {len(sentences)} sentences to file: {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(f"{sentence}{os.linesep}")


def create_extract_files(cli_input: CliInput, sentence_pairs: List[SentencePair]) -> None:
    logger.info("Creating extract files")
    if cli_input.output is None:
        logger.info("No output directory specified, defaulting to SIL_NLP_ENV.mt_corpora_dir")
        unique_dir = f"{cli_input.source_iso}-{cli_input.target_iso}-{cli_input.dataset_descriptor}-{time.strftime('%Y%m%d-%H%M%S')}"
        from ..common.environment import SIL_NLP_ENV

        output_dir = SIL_NLP_ENV.mt_corpora_dir / unique_dir
    else:
        logger.info("Using specified output directory")
        output_dir = Path(cli_input.output)

    logger.info(f"Outputting to directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def create_source_target_files(sub_sentence_pairs: List[SentencePair], suffix: str) -> None:
        logger.debug(f"Creating source and target files for suffix: '{suffix}'")

        def build_output_path(iso: str) -> Path:
            return output_dir / f"{iso}-{cli_input.dataset_descriptor}.{suffix}.txt"

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
    if cli_input.log_level is not None:
        log_level = getattr(logging, cli_input.log_level.upper())
        logger.setLevel(log_level)
        repair_logger.setLevel(log_level)
        clean_logger.setLevel(log_level)
    logger.info("Starting script")
    sentence_pairs = load_sentence_pairs(cli_input.input_file_path)
    create_extract_files(cli_input, sentence_pairs)
    logger.info("Completed script")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transforms XRI tsv files into extract files")

    parser.add_argument("input_file", help="Path to the input tsv file with UTF-8 encoding", type=str)
    parser.add_argument("source_iso", help="The ISO 693-3 code for the source/LWC language", type=str)
    parser.add_argument("target_iso", help="The ISO 693-3 code for the target/vernacular language", type=str)
    parser.add_argument("dataset", help="A descriptor of the dataset to be used in the output filename", type=str)
    parser.add_argument(
        "-output", help="Optional path to the output directory where extract files are generated", type=str
    )
    parser.add_argument(
        "-log_level", help="Optional parameter to override the default logging level for this script", type=str
    )
    args = parser.parse_args()

    cli_input = CliInput(
        input_file_path=args.input_file,
        source_iso=args.source_iso,
        target_iso=args.target_iso,
        dataset_descriptor=args.dataset,
        output=args.output,
        log_level=args.log_level,
    )
    run(cli_input)


if __name__ == "__main__":
    main()
