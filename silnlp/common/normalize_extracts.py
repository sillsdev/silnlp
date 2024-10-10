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
import regex

from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__package__ + ".normalize_extracts")
all_loggers = [logger]  # More to be added


class PunctuationCategory(Enum):
    left_clinging = "left_clinging"
    right_clinging = "right_clinging"
    left_right_clinging = "left_right_clinging"
    unclinging = "unclinging"


@dataclass(frozen=True)
class PunctuationNormalizationRule:
    character: str  # length 1
    category: PunctuationCategory


@dataclass(frozen=True)
class SentenceTransformation:
    """
    A representation of a delta applied to a sentence at a particular position
    """

    start_index: int
    end_index: int
    original: str
    replacement: str
    description: str


class WarningCode(Enum):
    multiple_punctuation = 0


@dataclass(frozen=True)
class NormalizationWarning:
    """
    A warning of a potential false negative/positive spotted during normalization
    For example it may notice a character that looks like punctuation,
    but it's not included in the list of punctuation characters specified by the user
    """

    start_index: int
    end_index: int
    warning_code: WarningCode
    description: str


@dataclass(frozen=True)
class SentenceNormalizationSummary:
    """
    A summary of the normalization process for a particular sentence.
    Warnings about potential issues are also included.
    """

    original_sentence: str
    normalized_sentence: str
    transformations: List[SentenceTransformation]
    warnings: List[NormalizationWarning]


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
    return [
        path
        for path in matching_paths
        if path.is_file() and path.suffix == ".txt" and not str(path).endswith("norm.txt")
    ]


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


class Normalizer:
    """
    Encapsulates the state required to normalize sentences
    """

    def __init__(self, punctuation_normalization_rules: List[PunctuationNormalizationRule]):
        self.punctuation_normalization_rules = punctuation_normalization_rules

        self.consecutive_spaces = regex.compile("\\s+")
        escaped_punctuation_chars = "".join(regex.escape(rule.character) for rule in punctuation_normalization_rules)
        self.punctuation_regex = regex.compile(f"[{escaped_punctuation_chars}\s]+")
        # TODO - ensure no duplicate characters
        # TODO - ensure strings are length 1
        # TODO - ensure no whitespace

    def normalize(self, sentence: str) -> SentenceNormalizationSummary:
        """
        Returns a normalized version of the input string.
        """
        logger.debug(f"Normalizing '{sentence}'")

        # TODO - just here for debugging, remove when code is mature
        def pretty_print_coords(start_index: int, end_index: int) -> None:
            logger.debug(sentence)
            logger.debug("012345678901234567890123456789")
            logger.debug("0         1         2         ")
            logger.debug(start_index * " " + len(text) * "*")
            logger.debug(start_index * " " + f"({start_index},{end_index})")

        punctuation_groups = ((match.group(), match.span()) for match in regex.finditer(self.punctuation_regex, sentence))

        # Categorize each group found
        consecutive_spaces_found: List[Tuple[str, int, int]] = []
        single_punctuation_found: List[Tuple[str, int, int]] = []
        multiple_punctuation_found: List[Tuple[str, int, int]] = []
        for text, (start_index, end_index) in punctuation_groups:
            if text == " ":
                continue
            pretty_print_coords(start_index, end_index)

            whitespace_removed = regex.sub(self.consecutive_spaces, "", text)
            # match is completely whitespace
            if not whitespace_removed:
                append_to = consecutive_spaces_found
            # match has one punctuation character
            elif len(whitespace_removed) == 1:
                append_to = single_punctuation_found
            # match has 2+ punctuation character
            else:
                append_to = multiple_punctuation_found

            append_to.append((text, start_index, end_index))

        consecutive_spaces_transformations = [
            SentenceTransformation(start_index, end_index, text, " ", "Consecutive whitespace found")
            for text, start_index, end_index in consecutive_spaces_found
        ]
        logger.debug(f"#consecutive space blocks   ={len(consecutive_spaces_found)}")
        logger.debug(f"#single punctuation chunks  ={len(single_punctuation_found)}")
        logger.debug(f"#multiple punctuation chunks={len(multiple_punctuation_found)}")

        # TODO - probably move this method out
        def normalize_single_punctuation_group(start_index: int, end_index: int, original_text: str) -> Optional[str]:
            """
            Normalizes the portion of text found that contains a single punctuation character
            If the normalized text is the same as the original text, None is returned to indicate no replacement is needed
            """
            punctuation_char = regex.sub(self.consecutive_spaces, "", original_text)
            punctuation_rule = next(
                filter(lambda rule: rule.character == punctuation_char, self.punctuation_normalization_rules)
            )
            # TODO - add checking around the boundary
            if punctuation_rule.category == PunctuationCategory.left_clinging:
                return " " + punctuation_char
            elif punctuation_rule.category == PunctuationCategory.right_clinging:
                return punctuation_char + " "
            elif punctuation_rule.category == PunctuationCategory.left_right_clinging:
                # Figure out if it's left or right clinging
                # TODO think about edge cases around the boundary
                # TODO in the case of no surrounding punctuation this implementation is left biased
                # TODO think about the case of space on both sides
                if original_text[0] == punctuation_char:
                    return " " + punctuation_char
                elif original_text[-1] == punctuation_char:
                    return " " + punctuation_char
                else:
                    # TODO - this needs to generate a warning
                    # Probably pick this up one higher up
                    # When those 3 lists are built, you could add more context
                    # Or you could do a dirty side effect
                    return None
            elif punctuation_rule.category == PunctuationCategory.unclinging:
                return " " + punctuation_char + " "
            else:
                return None

        single_punctuation_analysis = [
            (start_index, end_index, text, normalize_single_punctuation_group(start_index, end_index, text))
            for text, start_index, end_index in single_punctuation_found
        ]

        single_punctuation_transformations = [
            SentenceTransformation(start_index, end_index, text, normalized, "Punctuation found")
            for start_index, end_index, original, normalized in single_punctuation_analysis
            if normalized is not None
        ]

        multiple_punctuation_warnings = [
            NormalizationWarning(
                start_index,
                end_index,
                WarningCode.multiple_punctuation,
                f"Text {text} contains 2 or more punctuation characters - currently this is not normalized",
            )
            for text, start_index, end_index in multiple_punctuation_found
        ]

        # TODO - add other kinds of warnings
        all_warnings = multiple_punctuation_warnings

        all_transformations = sorted(
            consecutive_spaces_transformations + single_punctuation_transformations,
            key=lambda transformation: transformation.start_index,
        )

        # Rebuild the string by applying all the transformations
        # We extract the parts unaffected by normalization, then rebuild by interleaving them with the normalized parts
        # Example:
        #   Hello , there you  !Bye
        #   -----   ---------   ---     <-- original parts
        #        ---         ---        <-- parts to normalize
        #        ", "         "!"       <-- normalized replacements
        parts = []
        last_part_end_index = 0
        for transformation in all_transformations:
            # The part prior to this normalization segment
            parts.append(sentence[last_part_end_index : transformation.start_index])
            last_part_end_index = transformation.end_index
            parts.append(transformation.replacement)
        # Deal with the ending original part
        parts.append(sentence[last_part_end_index:])

        normalized = "".join(parts)
        # TODO - test an example where there's no normalization
        # TODO - shortcircuit this if no normalization? I guess it will be pretty fast

        return SentenceNormalizationSummary(
            original_sentence=sentence,
            normalized_sentence=normalized,
            transformations=all_transformations,
            warnings=all_warnings,
        )


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

    # TODO - replace with cli input
    normalizer = Normalizer(
        [
            PunctuationNormalizationRule(".", PunctuationCategory.right_clinging),
            PunctuationNormalizationRule(",", PunctuationCategory.right_clinging),
            PunctuationNormalizationRule("?", PunctuationCategory.right_clinging),
            PunctuationNormalizationRule("!", PunctuationCategory.right_clinging),
        ]
    )
    # TODO - test with left clinging
    # TODO - test with left/right clinging
    # TODO - test with unclinging

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
        normalized_lines: List[str] = [
            # TODO - process the summaries
            normalizer.normalize(extract_sentence).normalized_sentence
            for extract_sentence in input_lines
        ]
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
