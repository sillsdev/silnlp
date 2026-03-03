import argparse
import logging
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from math import exp
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from machine.scripture import ALL_BOOK_IDS, VerseRef
from scipy.stats import linregress

from ..common.translator import CONFIDENCE_SUFFIX, ConfidenceFile, TxtConfidenceFile, UsfmConfidenceFile
from .config import get_mt_exp_dir
from .test import VERSE_SCORES_SUFFIX

LOGGER = logging.getLogger(__package__ + ".quality_estimation")
CANONICAL_ORDER = {book: i for i, book in enumerate(ALL_BOOK_IDS)}


@dataclass
class Score:
    confidence: float
    projected_chrf3: float


@dataclass
class VerseScore(Score):
    vref: VerseRef

    @classmethod
    def get_scores_from_confidence_file(
        cls, confidence_file: UsfmConfidenceFile, slope: float, intercept: float
    ) -> List["VerseScore"]:
        verse_scores: List[VerseScore] = []
        for vref, confidence in confidence_file.verse_confidence_iterator():
            projected_chrf3 = slope * confidence + intercept
            verse_scores.append(cls(confidence, projected_chrf3, vref))
        return verse_scores


@dataclass
class ChapterScores:
    scores: Dict[str, Dict[int, Score]] = field(default_factory=lambda: defaultdict(dict))
    verse_usabilities: Dict[str, Dict[int, List[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    def add_score(self, book: str, chapter: int, score: Score) -> None:
        self.scores[book][chapter] = score

    def get_score(self, book: str, chapter: int) -> Optional[Score]:
        return self.scores.get(book, {}).get(chapter)

    def append_verse_usability(self, book: str, chapter: int, usability: float) -> None:
        self.verse_usabilities[book][chapter].append(usability)

    def get_verse_usabilities(self, book: str, chapter: int) -> List[float]:
        return self.verse_usabilities.get(book, {}).get(chapter, [])

    def add_scores_from_confidence_file(
        self, book: str, confidence_file: UsfmConfidenceFile, slope: float, intercept: float
    ) -> None:
        for chapter, confidence in confidence_file.chapter_confidence_iterator():
            projected_chrf3 = slope * confidence + intercept
            score = Score(confidence, projected_chrf3)
            self.add_score(book, chapter, score)


@dataclass
class BookScores:
    scores: Dict[str, Score] = field(default_factory=dict)
    verse_usabilities: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    seen_files: Set[Path] = field(default_factory=set)

    def add_score(self, book: str, score: Score) -> None:
        self.scores[book] = score

    def get_score(self, book: str) -> Optional[Score]:
        return self.scores.get(book)

    def append_verse_usability(self, book: str, usability: float) -> None:
        self.verse_usabilities[book].append(usability)

    def get_verse_usabilities(self, book: str) -> List[float]:
        return self.verse_usabilities.get(book, [])

    def add_scores_from_confidence_file(
        self, confidence_file: UsfmConfidenceFile, slope: float, intercept: float
    ) -> None:
        books_path = confidence_file.get_books_path()
        if books_path.is_file() and books_path not in self.seen_files:
            self.seen_files.add(books_path)
            for book, confidence in confidence_file.book_confidence_iterator():
                projected_chrf3 = slope * confidence + intercept
                score = Score(confidence, projected_chrf3)
                self.add_score(book, score)


@dataclass
class SequenceScore(Score):
    sequence_num: str
    trg_draft_file_stem: str

    @classmethod
    def get_scores_from_confidence_file(
        cls, confidence_file: TxtConfidenceFile, slope: float, intercept: float
    ) -> List["SequenceScore"]:
        trg_draft_file_stem = confidence_file.get_trg_draft_file_path().stem
        sequence_scores: List[SequenceScore] = []
        for sequence_num, confidence in confidence_file.verse_confidence_iterator():
            projected_chrf3 = slope * confidence + intercept
            sequence_scores.append(cls(confidence, projected_chrf3, sequence_num, trg_draft_file_stem))
        return sequence_scores


@dataclass
class TxtFileScores:
    scores: Dict[str, Score] = field(default_factory=dict)
    sequence_usabilities: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    seen_files: Set[Path] = field(default_factory=set)

    def add_score(self, trg_draft_file_stem: str, score: Score) -> None:
        self.scores[trg_draft_file_stem] = score

    def get_score(self, trg_draft_file_stem: str) -> Optional[Score]:
        return self.scores.get(trg_draft_file_stem)

    def append_sequence_usability(self, trg_draft_file_stem: str, usability: float) -> None:
        self.sequence_usabilities[trg_draft_file_stem].append(usability)

    def get_sequence_usabilities(self, trg_draft_file_stem: str) -> List[float]:
        return self.sequence_usabilities.get(trg_draft_file_stem, [])

    def add_scores_from_confidence_file(
        self, confidence_file: TxtConfidenceFile, slope: float, intercept: float
    ) -> None:
        files_path = confidence_file.get_files_path()
        if files_path.is_file() and files_path not in self.seen_files:
            self.seen_files.add(files_path)
            for trg_draft_file_stem, confidence in confidence_file.file_confidence_iterator():
                projected_chrf3 = slope * confidence + intercept
                score = Score(confidence, projected_chrf3)
                self.add_score(trg_draft_file_stem, score)


def estimate_quality(verse_test_scores_path: Path, confidence_file_paths: List[Path]) -> None:
    verse_test_scores_path, confidence_files = validate_inputs(verse_test_scores_path, confidence_file_paths)
    verse_scores, chapter_scores, book_scores, sequence_scores, txt_file_scores = project_chrf3(
        verse_test_scores_path, confidence_files
    )
    compute_usable_proportions(
        verse_scores,
        chapter_scores,
        book_scores,
        sequence_scores,
        txt_file_scores,
        confidence_files[0].get_path().parent,
    )


def validate_inputs(
    verse_test_scores_path: Path, confidence_file_paths: List[Path]
) -> Tuple[Path, List[ConfidenceFile]]:
    if not verse_test_scores_path.exists():
        raise FileNotFoundError(f"Test data file {verse_test_scores_path} does not exist.")
    elif verse_test_scores_path.is_dir():
        LOGGER.info(f"Searching for files with suffix {VERSE_SCORES_SUFFIX} in directory {verse_test_scores_path}.")
        test_files = list(verse_test_scores_path.glob(f"*{VERSE_SCORES_SUFFIX}"))
        if not test_files:
            raise ValueError(
                f"No test data file with the {VERSE_SCORES_SUFFIX} suffix found in directory {verse_test_scores_path}."
            )
        verse_test_scores_path = test_files[0]
        LOGGER.info(f"Using test data file {verse_test_scores_path}.")

    if len(confidence_file_paths) == 0:
        raise ValueError("At least one confidence file must be provided.")
    if not all(cf.is_file() for cf in confidence_file_paths):
        missing_files = [str(cf) for cf in confidence_file_paths if not cf.is_file()]
        raise FileNotFoundError(f"The following confidence files do not exist: {', '.join(missing_files)}")

    confidence_files: List[ConfidenceFile] = []
    for cf in confidence_file_paths:
        confidence_files.append(ConfidenceFile.from_confidence_file_path(cf))

    return verse_test_scores_path, confidence_files


def project_chrf3(
    verse_test_scores_path: Path, confidence_files: List[ConfidenceFile]
) -> Tuple[List[VerseScore], ChapterScores, BookScores, List[SequenceScore], TxtFileScores]:
    chrf3_scores, confidence_scores = extract_test_data(verse_test_scores_path)
    if len(chrf3_scores) != len(confidence_scores):
        raise ValueError(
            f"The number of chrF3 scores ({len(chrf3_scores)}) and confidence scores ({len(confidence_scores)}) "
            f"in {verse_test_scores_path} do not match."
        )
    slope, intercept = linregress(confidence_scores, chrf3_scores)[:2]
    verse_scores: List[VerseScore] = []
    chapter_scores: ChapterScores = ChapterScores()
    book_scores: BookScores = BookScores()
    sequence_scores: List[SequenceScore] = []
    txt_file_scores: TxtFileScores = TxtFileScores()
    for confidence_file in confidence_files:
        if isinstance(confidence_file, UsfmConfidenceFile):
            file_verse_scores = VerseScore.get_scores_from_confidence_file(confidence_file, slope, intercept)
            if not file_verse_scores:
                LOGGER.warning(f"No verse scores found in confidence file {confidence_file.get_path()}. Skipping.")
                continue
            verse_scores += file_verse_scores
            chapter_scores.add_scores_from_confidence_file(
                file_verse_scores[0].vref.book, confidence_file, slope, intercept
            )
            book_scores.add_scores_from_confidence_file(confidence_file, slope, intercept)
        elif isinstance(confidence_file, TxtConfidenceFile):
            file_sequence_scores = SequenceScore.get_scores_from_confidence_file(confidence_file, slope, intercept)
            if not file_sequence_scores:
                LOGGER.warning(f"No sequence scores found in confidence file {confidence_file.get_path()}. Skipping.")
                continue
            sequence_scores += file_sequence_scores
            txt_file_scores.add_scores_from_confidence_file(confidence_file, slope, intercept)
    return verse_scores, chapter_scores, book_scores, sequence_scores, txt_file_scores


def extract_test_data(verse_test_scores_path: Path) -> Tuple[List[float], List[float]]:
    chrf3_scores: List[float] = []
    confidence_scores: List[float] = []
    with open(verse_test_scores_path, "r", encoding="utf-8") as f:
        header = next(f).strip().lower().split("\t")
        try:
            chrf3_index = header.index("chrf3")
            confidence_index = header.index("confidence")
        except ValueError as e:
            raise ValueError(
                f"Could not find 'chrF3' and/or 'confidence' columns in header of {verse_test_scores_path}: {header}"
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
                    f"Error parsing line {line_num} in {verse_test_scores_path}: {line.strip()}"
                    f" (chrF3 index: {chrf3_index}, confidence index: {confidence_index})"
                ) from e

    return chrf3_scores, confidence_scores


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
    GREEN_THRESHOLD = 0.745
    YELLOW_THRESHOLD = 0.62


class ChapterThresholds(Thresholds):
    GREEN_THRESHOLD = 0.745
    YELLOW_THRESHOLD = 0.62


class VerseThresholds(Thresholds):
    GREEN_THRESHOLD = 0.745
    YELLOW_THRESHOLD = 0.62


def compute_usable_proportions(
    verse_scores: List[VerseScore],
    chapter_scores: ChapterScores,
    book_scores: BookScores,
    sequence_scores: List[SequenceScore],
    txt_file_scores: TxtFileScores,
    output_dir: Path,
) -> None:
    usable_params, unusable_params = parse_parameters(output_dir / "usability_parameters.tsv")

    if verse_scores:
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

                chapter_scores.append_verse_usability(vref.book, int(vref.chapter), prob)
                book_scores.append_verse_usability(vref.book, prob)

                verse_file.write(
                    f"{vref.book}\t{vref.chapter_num}\t{vref.verse_num}\t{verse_score.projected_chrf3:.2f}\t{prob:.3f}\t{label}\n"
                )
        compute_chapter_usability(chapter_scores, output_dir)
        compute_book_usability(book_scores, output_dir)
    if sequence_scores:
        with open(output_dir / "usability_sequences.tsv", "w", encoding="utf-8", newline="\n") as sequence_file:
            sequence_file.write("Trg Draft File\tSequence Number\tProjected chrF3\tUsability\tLabel\n")
            for sequence_score in sequence_scores:
                if sequence_score.projected_chrf3 is None:
                    LOGGER.warning(f"Sequence {sequence_score.sequence_num} does not have a projected chrf3. Skipping.")
                    continue

                prob = calculate_usable_prob(sequence_score.projected_chrf3, usable_params, unusable_params)
                label = VerseThresholds.return_label(prob)

                txt_file_scores.append_sequence_usability(sequence_score.trg_draft_file_stem, prob)

                sequence_file.write(
                    f"{sequence_score.trg_draft_file_stem}\t{sequence_score.sequence_num}\t"
                    f"{sequence_score.projected_chrf3:.2f}\t{prob:.3f}\t{label}\n"
                )
        compute_txt_file_usability(txt_file_scores, output_dir)


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


def compute_chapter_usability(
    chapter_scores: ChapterScores,
    output_dir: Path,
) -> None:
    with open(output_dir / "usability_chapters.tsv", "w", encoding="utf-8", newline="\n") as chapter_file:
        chapter_file.write("Book\tChapter\tProjected chrF3\tUsability\tLabel\n")
        for book in sorted(chapter_scores.scores, key=lambda b: CANONICAL_ORDER[b]):
            for chapter in sorted(chapter_scores.scores[book]):
                chapter_usabilities = chapter_scores.get_verse_usabilities(book, chapter)
                avg_prob = sum(chapter_usabilities) / len(chapter_usabilities)
                label = ChapterThresholds.return_label(avg_prob)
                if not chapter_scores.get_score(book, chapter):
                    LOGGER.warning(f"{book} {chapter} does not have a projected chrf3.")
                    chapter_file.write(f"{book}\t{chapter}\t\t{avg_prob:.3f}\t{label}\n")
                    continue
                projected_chrf3 = chapter_scores.get_score(book, chapter).projected_chrf3
                chapter_file.write(f"{book}\t{chapter}\t{projected_chrf3:.2f}\t{avg_prob:.3f}\t{label}\n")


def compute_book_usability(
    book_scores: BookScores,
    output_dir: Path,
) -> None:
    with open(output_dir / "usability_books.tsv", "w", encoding="utf-8", newline="\n") as book_file:
        book_file.write("Book\tProjected chrF3\tUsability\tLabel\n")
        for book in sorted(book_scores.scores, key=lambda b: CANONICAL_ORDER[b]):
            # book/chapter usabilties are calculated from verse avg, not from book/chapter projected chrf3
            book_usabilities = book_scores.get_verse_usabilities(book)
            avg_prob = sum(book_usabilities) / len(book_usabilities)
            label = BookThresholds.return_label(avg_prob)
            if not book_scores.get_score(book):
                LOGGER.warning(f"{book} does not have a projected chrf3.")
                book_file.write(f"{book}\t\t{avg_prob:.3f}\t{label}\n")
                continue
            projected_chrf3 = book_scores.get_score(book).projected_chrf3
            book_file.write(f"{book}\t{projected_chrf3:.2f}\t{avg_prob:.3f}\t{label}\n")


def compute_txt_file_usability(
    txt_file_scores: TxtFileScores,
    output_dir: Path,
) -> None:
    with open(output_dir / "usability_txt_files.tsv", "w", encoding="utf-8", newline="\n") as txt_file:
        txt_file.write("Trg Draft File\tProjected chrF3\tUsability\tLabel\n")
        for trg_draft_file_stem in sorted(txt_file_scores.scores):
            txt_file_usabilities = txt_file_scores.get_sequence_usabilities(trg_draft_file_stem)
            avg_prob = sum(txt_file_usabilities) / len(txt_file_usabilities)
            label = BookThresholds.return_label(avg_prob)
            if not txt_file_scores.get_score(trg_draft_file_stem):
                LOGGER.warning(f"{trg_draft_file_stem} does not have a projected chrf3.")
                txt_file.write(f"{trg_draft_file_stem}\t\t{avg_prob:.3f}\t{label}\n")
                continue
            projected_chrf3 = txt_file_scores.get_score(trg_draft_file_stem).projected_chrf3
            txt_file.write(f"{trg_draft_file_stem}\t{projected_chrf3:.2f}\t{avg_prob:.3f}\t{label}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the quality of drafts created by an NMT model.")
    parser.add_argument(
        "verse_test_scores_file",
        type=str,
        help="Path relative to MT/experiments to a verse-level test score file, which is used to find line of best fit "
        + "for confidence and chrF3, e.g., project_folder/exp_folder/test.trg-predictions.detok.txt.5000.scores.tsv. "
        + "If a directory is provided instead, the first *.scores.tsv match is used.",
    )
    parser.add_argument(
        "confidence_files",
        nargs="*",
        type=str,
        help="Zero or more confidence file paths (.confidences.tsv suffix, e.g., "
        + "project_folder/exp_folder/infer/5000/source/631JN.SFM.confidences.tsv'). Paths are relative to "
        + "MT/experiments by default or to MT/experiments/--confidence-dir if --confidence-dir is specified. "
        + "Ignored when --books is used. If zero paths are provided and --books is not specified, "
        + "confidence files are auto detected in the --confidence-dir.",
    )
    parser.add_argument(
        "--confidence-dir",
        type=str,
        default=None,
        help="Directory relative to MT/experiments containing confidence files. "
        + "Required when using --books or when auto-detecting confidence files.",
    )
    parser.add_argument(
        "--books",
        nargs="+",
        metavar="book_ids",
        help="Provide book ids (e.g. 1JN LUK) to select confidence files rather than providing file paths with "
        + "the confidence_files positional argument.",
    )
    parser.add_argument(
        "--draft-index",
        type=int,
        default=None,
        help="If using --books with multiple drafts, optionally specify the draft index.",
    )
    args = parser.parse_args()

    using_files = bool(args.confidence_files)
    using_books = bool(args.books)
    using_auto_detect = not using_files and not using_books

    if using_files and using_books:
        raise ValueError("Specify either confidence_files or --books, not both.")

    if (using_books or using_auto_detect) and args.confidence_dir is None:
        raise ValueError("When using --books or auto-detecting confidence files, --confidence-dir must be specified.")
    confidence_dir = get_mt_exp_dir(args.confidence_dir or "")
    if not confidence_dir.is_dir():
        raise ValueError(f"Confidence directory {confidence_dir} does not exist or is not a directory.")

    if using_auto_detect:
        LOGGER.info(f"Auto-detecting confidence files in directory {confidence_dir}")
        confidence_file_paths = list(confidence_dir.glob(f"*{CONFIDENCE_SUFFIX}"))
    elif using_files:
        if len(args.confidence_files) == 0:
            raise ValueError("Please provide at least one confidence file for the confidence_files argument.")
        confidence_file_paths = [confidence_dir / confidence_file for confidence_file in args.confidence_files]
    elif using_books:
        if len(args.books) == 0:
            raise ValueError("Please provide at least one book for the --books argument.")
        if args.draft_index is not None:
            if not isinstance(args.draft_index, int) or args.draft_index < 0:
                raise ValueError("Draft index must be a non-negative integer.")
            draft_suffix = "." + str(args.draft_index)
        else:
            draft_suffix = ""
        confidence_file_paths = []
        for book_id in args.books:
            confidence_file_paths.extend(confidence_dir.glob(f"[0-9]*{book_id}{draft_suffix}.*{CONFIDENCE_SUFFIX}"))

    estimate_quality(get_mt_exp_dir(args.verse_test_scores_file), confidence_file_paths)


if __name__ == "__main__":
    main()
