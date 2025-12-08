import argparse
import logging
import re
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import List, Optional, Tuple

from machine.scripture import ALL_BOOK_IDS, VerseRef
from openpyxl import load_workbook
from scipy.stats import linregress

from .config import get_mt_exp_dir

LOGGER = logging.getLogger(__package__ + ".quality_estimation")
CANONICAL_ORDER = {book: i for i, book in enumerate(ALL_BOOK_IDS)}


@dataclass
class VerseScore:
    vref: VerseRef
    confidence: float
    projected_chrf3: Optional[float] = None


def estimate_quality(diff_predictions_file: Path, confidence_files: List[Path]) -> None:
    verse_scores: List[VerseScore] = project_chrf3(diff_predictions_file, confidence_files)
    compute_usable_proportions(verse_scores, confidence_files[0].parent)


def project_chrf3(diff_predictions_file: Path, confidence_files: List[Path]) -> List[VerseScore]:
    chrf3_scores, confidence_scores = extract_diff_predictions(diff_predictions_file)
    if len(chrf3_scores) != len(confidence_scores):
        raise ValueError(
            f"The number of chrF3 scores ({len(chrf3_scores)}) and confidence scores ({len(confidence_scores)}) "
            f"in {diff_predictions_file} do not match."
        )
    slope, intercept = linregress(confidence_scores, chrf3_scores)[:2]
    verse_scores: List[VerseScore] = []
    for confidence_file in confidence_files:
        file_scores = extract_confidences(confidence_file)
        verse_scores += file_scores
        with open(confidence_file.with_suffix(".projected_chrf3.tsv"), "w", encoding="utf-8") as output_file:
            output_file.write("VRef\tConfidence\tProjected chrF3\n")
            for verse_score in verse_scores:
                projected_chrf3 = slope * verse_score.confidence + intercept
                verse_score.projected_chrf3 = projected_chrf3
                output_file.write(f"{verse_score.vref}\t{verse_score.confidence}\t{projected_chrf3:.2f}\n")
    return verse_scores


def extract_diff_predictions(diff_predictions_file_path) -> Tuple[List[float], List[float]]:
    chrf3_scores = extract_diff_predictions_column(diff_predictions_file_path, "chrf3")
    confidence_scores = extract_diff_predictions_column(diff_predictions_file_path, "confidence")
    return chrf3_scores, confidence_scores


def extract_diff_predictions_column(file_path: Path, target_header: str) -> List[float]:
    wb = load_workbook(file_path)
    ws = wb.active

    header_row_idx = None
    col_idx = None

    for row in ws.iter_rows():
        for cell in row:
            if cell.value == "Score Summary":
                break
            if str(cell.value).lower() == target_header.lower():
                header_row_idx = cell.row
                col_idx = cell.column
                break
        if header_row_idx:
            break

    if not header_row_idx:
        raise ValueError(f"Header '{target_header}' not found.")

    data = []
    for row in ws.iter_rows(min_row=header_row_idx + 1, min_col=col_idx, max_col=col_idx):
        cell_value = row[0].value
        if cell_value is not None:
            data.append(float(cell_value))

    return data


def extract_confidences(input_file_path: Path) -> List[VerseScore]:
    current_book = ""
    current_chapter = 0
    current_verse = 0
    is_at_verse_reference = False

    vref_confidences: List[VerseScore] = []
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.lower().startswith("vref") or line.lower().startswith("sequence score"):
                continue

            match = re.match(r"^([0-9A-Z][A-Z]{2}) (\d+):(\d+)(/.*)?", line)
            if match:
                current_book = match.group(1)
                current_chapter = int(match.group(2))
                current_verse = int(match.group(3))
                extra = match.group(4)

                is_at_verse_reference = current_verse != 0 and not extra
            elif is_at_verse_reference:
                cols = line.split("\t")
                if cols:
                    vref_confidences += [
                        VerseScore(
                            VerseRef.from_string(f"{current_book} {current_chapter}:{current_verse}"), float(cols[0])
                        )
                    ]
    return vref_confidences


@dataclass
class UsabilityParameters:
    count: float
    mean: float
    variance: float


class Thresholds(ABC):
    GREEN_THRESHOLD: float
    YELLOW_THRESHOLD: float
    GREEN_LABEL = "Green"
    YELLOW_LABEL = "Yellow"
    RED_LABEL = "Red"

    @classmethod
    def return_label(cls, prob: float) -> str:
        if prob >= cls.GREEN_THRESHOLD:
            return cls.GREEN_LABEL
        elif prob >= cls.YELLOW_THRESHOLD:
            return cls.YELLOW_LABEL
        else:
            return cls.RED_LABEL


class BookThresholds(Thresholds):
    GREEN_THRESHOLD = 0.6
    YELLOW_THRESHOLD = 0.5


class ChapterThresholds(Thresholds):
    GREEN_THRESHOLD = 0.6
    YELLOW_THRESHOLD = 0.3


class VerseThresholds(Thresholds):
    GREEN_THRESHOLD = 0.6
    YELLOW_THRESHOLD = 0.3


def compute_usable_proportions(verse_scores: List[VerseScore], output_dir: Path) -> None:
    usable_params, unusable_params = parse_parameters(output_dir / "usability_parameters.tsv")

    book_totals = defaultdict(float)
    book_counts = defaultdict(int)
    chapter_totals = defaultdict(lambda: defaultdict(float))
    chapter_counts = defaultdict(lambda: defaultdict(int))

    with open(output_dir / "usability_verses.tsv", "w", encoding="utf-8", newline="\n") as verse_file:
        verse_file.write("Book\tChapter\tVerse\tUsability\tLabel\n")
        for verse_score in verse_scores:
            vref = verse_score.vref
            if vref.verse_num == 0:
                continue
            if verse_score.projected_chrf3 is None:
                LOGGER.warning(f"{vref} does not have a projected chrf3. Skipping.")
                continue

            prob = calculate_usable_prob(verse_score.projected_chrf3, usable_params, unusable_params)
            label = VerseThresholds.return_label(prob)

            book_totals[vref.book] += prob
            book_counts[vref.book] += 1
            chapter_totals[vref.book][vref.chapter_num] += prob
            chapter_counts[vref.book][vref.chapter_num] += 1

            verse_file.write(f"{vref.book}\t{vref.chapter_num}\t{vref.verse_num}\t{prob:.6f}\t{label}\n")

    with open(output_dir / "usability_books.tsv", "w", encoding="utf-8", newline="\n") as book_file:
        book_file.write("Book\tUsability\tLabel\n")
        for book in sorted(book_totals, key=lambda b: CANONICAL_ORDER[b]):
            avg_prob = book_totals[book] / book_counts[book]
            label = BookThresholds.return_label(avg_prob)
            book_file.write(f"{book}\t{avg_prob:.6f}\t{label}\n")

    with open(output_dir / "usability_chapters.tsv", "w", encoding="utf-8", newline="\n") as chapter_file:
        chapter_file.write("Book\tChapter\tUsability\tLabel\n")
        for book in sorted(chapter_totals, key=lambda b: CANONICAL_ORDER[b]):
            for chapter in sorted(chapter_totals[book]):
                avg_prob = chapter_totals[book][chapter] / chapter_counts[book][chapter]
                label = ChapterThresholds.return_label(avg_prob)
                chapter_file.write(f"{book}\t{chapter}\t{avg_prob:.6f}\t{label}\n")


def parse_parameters(parameter_file: Path) -> Tuple[UsabilityParameters, UsabilityParameters]:
    params = {
        "usable": UsabilityParameters(263, 51.4, 95.19),
        "unusable": UsabilityParameters(97, 45.85, 99.91),
    }
    if parameter_file.exists():
        with open(parameter_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    raise ValueError(
                        f"Malformed line {line_num} in {parameter_file}: expected 4 tab-separated columns, "
                        f"got {len(parts)}. Line content: {line.strip()}"
                    )
                label, count, mean, variance = parts
                params[label] = UsabilityParameters(float(count), float(mean), float(variance))
    else:
        LOGGER.warning(f"{parameter_file} does not exist. Using default parameters.")

    return params["usable"], params["unusable"]


def calculate_usable_prob(
    chrf3: float,
    usable: UsabilityParameters,
    unusable: UsabilityParameters,
) -> float:
    usable_weight = exp(-((chrf3 - usable.mean) ** 2) / (2 * usable.variance)) * usable.count
    unusable_weight = exp(-((chrf3 - unusable.mean) ** 2) / (2 * unusable.variance)) * unusable.count

    return usable_weight / (usable_weight + unusable_weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the quality of drafts created by an NMT model.")
    parser.add_argument(
        "diff_predictions",
        help="The diff predictions path relative to MT/experiments to determine line of best fit."
        + " e.g. 'project_folder/exp_folder/diff_predictions.5000.xlsx'.",
    )
    parser.add_argument(
        "confidence_files",
        nargs="*",
        type=Path,
        help="Relative paths for the confidence files to process (relative to MT/experiments or --confidence-dir "
        + "if specified) e.g. 'project_folder/exp_folder/infer/5000/source/631JN.SFM.confidences.tsv' or "
        + "'631JN.SFM.confidences.tsv --confidence-dir project_folder/exp_folder/infer/5000/source'.",
    )
    parser.add_argument(
        "--confidence-dir",
        type=Path,
        default=None,
        help="Folder (relative to experiment MT/experiments) containing confidence files e.g. 'infer/5000/source/'.",
    )
    parser.add_argument(
        "--books",
        nargs="+",
        metavar="book_ids",
        help="Provide book ids (e.g. 1JN LUK) to select confidence files rather than providing file paths with "
        + "the confidence_files positional argument.",
    )
    parser.add_argument(
        "--draft-index", type=int, default=None, help="If using --books with multiple drafts, specify the draft index."
    )
    args = parser.parse_args()

    using_files = bool(args.confidence_files)
    using_books = bool(args.books)

    if using_files and using_books:
        raise ValueError("Specify either confidence_files or --books, not both.")
    if not using_files and not using_books:
        raise ValueError(
            "You must specify either confidence_files or --books to indicate which confidence files to use."
        )

    confidence_dir = get_mt_exp_dir(args.confidence_dir or Path())

    if using_files:
        if len(args.confidence_files) == 0:
            raise ValueError("Please provide at least one confidence file for the confidence_files argument.")
        confidence_files = [
            confidence_dir / confidence_file if confidence_dir else confidence_file
            for confidence_file in args.confidence_files
        ]

    elif using_books:
        if len(args.books) == 0:
            raise ValueError("Please provide at least one book for the --books argument.")
        if args.draft_index is not None:
            if not isinstance(args.draft_index, int) or args.draft_index < 0:
                raise ValueError("Draft index must be a non-negative integer.")
            draft_suffix = "." + str(args.draft_index)
        else:
            draft_suffix = ""
        confidence_files = []
        for book_id in args.books:
            confidence_files.extend(confidence_dir.glob(f"[0-9]*{book_id}{draft_suffix}.*.confidences.tsv"))

    estimate_quality(get_mt_exp_dir(args.diff_predictions), confidence_files)


if __name__ == "__main__":
    main()
