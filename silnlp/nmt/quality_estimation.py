import argparse
import logging
import re
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from math import exp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from machine.scripture import ALL_BOOK_IDS, VerseRef
from openpyxl import load_workbook
from scipy.stats import linregress

from .config import get_mt_exp_dir

LOGGER = logging.getLogger(__package__ + ".quality_estimation")
CANONICAL_ORDER = {book: i for i, book in enumerate(ALL_BOOK_IDS)}


@dataclass
class Score:
    confidence: float
    projected_chrf3: float


@dataclass
class VerseScore(Score):
    vref: VerseRef


@dataclass
class ChapterScores:
    scores: Dict[str, Dict[int, Score]] = field(default_factory=lambda: defaultdict(dict))

    def add_score(self, book: str, chapter: int, score: Score) -> None:
        self.scores[book][chapter] = score

    def get_score(self, book: str, chapter: int) -> Optional[Score]:
        return self.scores.get(book, {}).get(chapter)


@dataclass
class BookScores:
    scores: Dict[str, Score] = field(default_factory=dict)  # book -> Score

    def add_score(self, book: str, score: Score) -> None:
        self.scores[book] = score

    def get_score(self, book: str) -> Optional[Score]:
        return self.scores.get(book)


def estimate_quality(test_data_path: Path, confidence_files: List[Path]) -> None:
    validate_inputs(test_data_path, confidence_files)
    verse_scores, chapter_scores, book_scores = project_chrf3(test_data_path, confidence_files)
    compute_usable_proportions(verse_scores, chapter_scores, book_scores, confidence_files[0].parent)


def validate_inputs(test_data_path: Path, confidence_files: List[Path]) -> None:
    if test_data_path is None:
        raise ValueError("Test data file path must be provided.")
    if confidence_files is None or len(confidence_files) == 0:
        raise ValueError("At least one confidence file must be provided.")
    if not test_data_path.is_file():
        raise FileNotFoundError(f"Test data file {test_data_path} does not exist.")
    if not all(cf.is_file() for cf in confidence_files):
        missing_files = [str(cf) for cf in confidence_files if not cf.is_file()]
        raise FileNotFoundError(f"The following confidence files do not exist: {', '.join(missing_files)}")


def project_chrf3(
    test_data_path: Path, confidence_files: List[Path]
) -> Tuple[List[VerseScore], ChapterScores, BookScores]:
    chrf3_scores, confidence_scores = extract_test_data(test_data_path)
    if len(chrf3_scores) != len(confidence_scores):
        raise ValueError(
            f"The number of chrF3 scores ({len(chrf3_scores)}) and confidence scores ({len(confidence_scores)}) "
            f"in {test_data_path} do not match."
        )
    slope, intercept = linregress(confidence_scores, chrf3_scores)[:2]
    verse_scores: List[VerseScore] = []
    chapter_scores: ChapterScores = ChapterScores()
    book_scores: BookScores = BookScores()
    for confidence_file in confidence_files:
        file_scores = get_verse_scores(confidence_file, slope, intercept)
        book = file_scores[0].vref.book if file_scores else None
        verse_scores += file_scores
        if confidence_file.with_suffix(".chapters.tsv").is_file():
            with open(confidence_file.with_suffix(".chapters.tsv"), "r", encoding="utf-8") as chapter_file:
                next(chapter_file)
                for line in chapter_file:
                    cols = line.strip().split("\t")
                    chapter = int(cols[0])
                    confidence = float(cols[1])
                    projected_chrf3 = slope * confidence + intercept
                    score = Score(confidence, projected_chrf3)
                    chapter_scores.add_score(book, chapter, score)
    if (confidence_files[0].parent / "confidences.books.tsv").is_file():
        with open(confidence_files[0].parent / "confidences.books.tsv", "r", encoding="utf-8") as book_file:
            next(book_file)
            for line in book_file:
                cols = line.strip().split("\t")
                book = cols[0]
                confidence = float(cols[1])
                projected_chrf3 = slope * confidence + intercept
                score = Score(confidence, projected_chrf3)
                book_scores.add_score(book, score)
    return verse_scores, chapter_scores, book_scores


def extract_test_data(test_data_path: Path) -> Tuple[List[float], List[float]]:
    chrf3_scores: List[float] = []
    confidence_scores: List[float] = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        header = next(f).strip().lower().split("\t")
        try:
            chrf3_index = header.index("chrf3")
            confidence_index = header.index("confidence")
        except ValueError as e:
            raise ValueError(
                f"Could not find 'chrF3' and 'confidence' columns in header of {test_data_path}: {header}"
            ) from e
        for line_num, line in enumerate(f, start=2):
            cols = line.strip().split("\t")
            try:
                chrf3 = float(cols[chrf3_index])
                confidence = float(cols[confidence_index])
                chrf3_scores.append(chrf3)
                confidence_scores.append(confidence)
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Error parsing line {line_num} in {test_data_path}: {line.strip()}"
                    f" (chrF3 index: {chrf3_index}, confidence index: {confidence_index})"
                ) from e

    return chrf3_scores, confidence_scores


def get_verse_scores(input_file_path: Path, slope: float, intercept: float) -> List[VerseScore]:
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
                    confidence = float(cols[0])
                    projected_chrf3 = slope * confidence + intercept
                    vref_confidences += [
                        VerseScore(
                            confidence,
                            projected_chrf3,
                            VerseRef.from_string(f"{current_book} {current_chapter}:{current_verse}"),
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
    GREEN_THRESHOLD = 0.687
    YELLOW_THRESHOLD = 0.62


class ChapterThresholds(Thresholds):
    GREEN_THRESHOLD = 0.687
    YELLOW_THRESHOLD = 0.62


class VerseThresholds(Thresholds):
    GREEN_THRESHOLD = 0.687
    YELLOW_THRESHOLD = 0.62


def compute_usable_proportions(
    verse_scores: List[VerseScore], chapter_scores: ChapterScores, book_scores: BookScores, output_dir: Path
) -> None:
    usable_params, unusable_params = parse_parameters(output_dir / "usability_parameters.tsv")

    book_totals = defaultdict(float)
    book_counts = defaultdict(int)
    chapter_totals = defaultdict(lambda: defaultdict(float))
    chapter_counts = defaultdict(lambda: defaultdict(int))

    with open(output_dir / "usability_verses.tsv", "w", encoding="utf-8", newline="\n") as verse_file:
        verse_file.write("Book\tChapter\tVerse\tProjected chrF3\tUsability\tLabel\n")
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

            verse_file.write(
                f"{vref.book}\t{vref.chapter_num}\t{vref.verse_num}\t{verse_score.projected_chrf3:.6f}\t{prob:.6f}\t{label}\n"
            )

    with open(output_dir / "usability_books.tsv", "w", encoding="utf-8", newline="\n") as book_file:
        book_file.write("Book\tProjected chrF3\tUsability\tLabel\n")
        for book in sorted(book_totals, key=lambda b: CANONICAL_ORDER[b]):
            # book/chapter usabilties are calculated from verse avg, not from book/chapter projected chrf3
            avg_prob = book_totals[book] / book_counts[book]
            label = BookThresholds.return_label(avg_prob)
            if not book_scores.get_score(book):
                LOGGER.warning(f"{book} does not have a projected chrf3.")
                book_file.write(f"{book}\t\t{avg_prob:.6f}\t{label}\n")
                continue
            projected_chrf3 = book_scores.get_score(book).projected_chrf3
            book_file.write(f"{book}\t{projected_chrf3:.2f}\t{avg_prob:.6f}\t{label}\n")

    with open(output_dir / "usability_chapters.tsv", "w", encoding="utf-8", newline="\n") as chapter_file:
        chapter_file.write("Book\tChapter\tProjected chrF3\tUsability\tLabel\n")
        for book in sorted(chapter_totals, key=lambda b: CANONICAL_ORDER[b]):
            for chapter in sorted(chapter_totals[book]):
                avg_prob = chapter_totals[book][chapter] / chapter_counts[book][chapter]
                label = ChapterThresholds.return_label(avg_prob)
                if not chapter_scores.get_score(book, chapter):
                    LOGGER.warning(f"{book} {chapter} does not have a projected chrf3.")
                    chapter_file.write(f"{book}\t{chapter}\t\t{avg_prob:.6f}\t{label}\n")
                    continue
                projected_chrf3 = chapter_scores.get_score(book, chapter).projected_chrf3
                chapter_file.write(f"{book}\t{chapter}\t{projected_chrf3:.2f}\t{avg_prob:.6f}\t{label}\n")


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
        "test_data_file",
        type=str,
        help="The tsv file relative to MT/experiments containing the test data to determine line of best fit."
        + "e.g. `project_folder/exp_folder/test.trg-predictions.detok.txt.5000.scores.tsv`",
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

    estimate_quality(get_mt_exp_dir(args.test_data_file), confidence_files)


if __name__ == "__main__":
    main()
