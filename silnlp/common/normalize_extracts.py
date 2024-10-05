"""
This script is passed a directory containing extract files.
It loads the extract files and normalizes them, then writes them to a new location.

Normalization currently entails simple whitespace changes.
See https://github.com/sillsdev/silnlp/issues/494 for more context.

Example:

    $ python -m silnlp.common.normalize_extracts input_dir --output output_dir

Suppose the input dir had the files below. The output dir will contain the normalized equivalents of those
files with `.norm` put before `.txt` in each filename.

      input_dir                                    output_dir
        ├── swa-extract.all.txt                      ├── swa-extract.all.norm.txt
        ├── ngq-extract.all.txt                      ├── ngq-extract.all.norm.txt
        ├── swa-extract.train.txt     normalize      ├── swa-extract.train.norm.txt
        ├── ngq-extract.train.txt     -------->      ├── ngq-extract.train.norm.txt
        ├── swa-extract.val.txt                      ├── swa-extract.val.norm.txt
        ├── ngq-extract.val.txt                      ├── ngq-extract.val.norm.txt
        ├── swa-extract.test.txt                     ├── swa-extract.test.norm.txt
        └── ngq-extract.test.txt                     └── ngq-extract.test.norm.txt
                                                                          ^^^^

The output dir is optional. When not specified, the input_dir is used.

Only files in the input_dir that have `.txt` extension (and don't end with `.norm.txt`) will be considered.
An optional glob filter can be added to further reduce which input files are transformed, using `--filter GLOB`.

If an output file already exists in the output directory, it won't be written over and an error will be logged.
The optional `--overwrite` flag will bypass this.

By default the script uses the logging configuration inherited from the parent packages (which should log at INFO level).
You can change the logging level with the optional `--log-level LOG_LEVEL` which accepts values like:
"DEBUG", "INFO", "WARNING/WARN", "ERROR" and "CRITICAL".
"""

import argparse
import logging
import os

from dataclasses import dataclass

from glob import glob
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__package__ + ".normalize_extracts")
all_loggers = [logger]  # More to be added


@dataclass(frozen=True)
class CliInput:
    input_dir: str
    output_dir: Optional[str]
    overwrite: bool
    filter: Optional[str]
    log_level: Optional[str]


def get_files_to_normalize(input_dir: Path, filter: Optional[str]) -> List[Path]:
    """
    Searches the top level of the input directory for extract files
    that aren't normalized.
    If the filter is defined, then further filtering of those candidates is performed.
    """
    if filter is None:
        logger.debug(f"Searching files in input dir: '{input_dir}'")
        matching_filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    else:
        logger.debug(f"Searching files in input dir: '{input_dir}' that satisfy glob '{filter}'")
        matching_filenames = glob(os.path.join(input_dir, filter), recursive=False)

    matching_paths: List[Path] = [Path(f) for f in matching_filenames]
    return [path for path in matching_paths if path.is_file() and path.suffix == ".txt" and not str(path).endswith("norm.txt")]


def normalized_path(output_dir: Path, input_path: Path) -> Path:
    """
    Uses the input path to generate corresponding output path with "norm" in the name.
    e.g. extract.all.txt -> extract.all.norm.txt
    """
    input_filename = input_path.parts[-1]
    output_filename_parts = input_filename.split(".")[0:-1]
    output_filename_parts.append("norm")
    output_filename_parts.append("txt")
    output_filename = ".".join(output_filename_parts)
    return output_dir / output_filename


def normalize(extract_sentence: str) -> str:
    """
    Returns a normalized version of the input string.
    """
    # TODO
    return extract_sentence


def load_extract_file(path: Path) -> List[str]:
    with open(path, "r", encoding="UTF-8") as file:
        return [line.rstrip() for line in file]


def write_extract_file(path: Path, sentences: List[str]) -> None:
    logger.debug(f"Writing {len(sentences)} sentences to file: {path}")
    with open(path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(f"{sentence}\n")


def run(cli_input: CliInput) -> None:
    if cli_input.log_level is not None:
        log_level = getattr(logging, cli_input.log_level.upper())
        for log in all_loggers:
            log.setLevel(log_level)

    logger.info("Starting script")

    input_dir = Path(cli_input.input_dir)

    if cli_input.output_dir is not None:
        output_dir = Path(cli_input.output_dir)
    else:
        output_dir = input_dir
    logger.info(f"Output dir set to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_normalize: List[Path] = get_files_to_normalize(input_dir, cli_input.filter)
    logger.info(f"Found {len(files_to_normalize)} files to normalize")

    for input_path in files_to_normalize:
        logger.debug(f"Processing file {input_path}")
        output_path: Path = normalized_path(output_dir, input_path)
        logger.debug(f"Outputting to {output_path}")
        if output_path.is_file() and not cli_input.overwrite:
            logger.error(
                f"Outpath '{output_path}' already exists. Skipping input {input_path}. "
                + "You can use the --overwrite flag to write over existing files."
            )
            continue

        input_lines: List[str] = load_extract_file(input_path)
        logger.debug(f"Found {len(input_lines)} lines in file")
        normalized_lines: List[str] = [normalize(extract_sentence) for extract_sentence in input_lines]
        write_extract_file(output_path, normalized_lines)
        logger.debug(f"Finished processing {input_path}")

    logger.info("Completed script")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalizes extract files")

    parser.add_argument(
        "input_dir", help="Path to the directory containing the extract files to be normalized", type=str
    )
    parser.add_argument(
        "--output-dir",
        help="Optional path to the output directory where the normalized extract files will be dumped. "
        + "When not specified the input directory is used",
        type=str,
    )
    parser.add_argument(
        "--overwrite",
        help="Optional parameter to make output files overwrite existing files of the same name",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--filter",
        help="Optional glob filter to narrow down the transformed files to only those those that satisfy the glob",
        type=str,
    )
    parser.add_argument(
        "--log-level", help="Optional parameter to override the default logging level for this script", type=str
    )
    args = parser.parse_args()

    cli_input = CliInput(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        filter=args.filter,
        log_level=args.log_level,
    )
    run(cli_input)


if __name__ == "__main__":
    main()
