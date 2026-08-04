import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from machine.scripture import ALL_BOOK_IDS, VerseRef

from ..common.environment import SilNlpEnv
from ..common.linear_regression import LinearRegressionResult
from ..common.translator import CONFIDENCE_SUFFIX, ConfidenceFile, TxtConfidenceFile, UsfmConfidenceFile
from .test import LINREGRESS_PREFIX

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

    def add_score(self, book: str, chapter: int, score: Score) -> None:
        self.scores[book][chapter] = score

    def get_score(self, book: str, chapter: int) -> Optional[Score]:
        return self.scores.get(book, {}).get(chapter)

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

    def add_score(self, book: str, score: Score) -> None:
        self.scores[book] = score

    def get_score(self, book: str) -> Optional[Score]:
        return self.scores.get(book)

    def add_scores_from_confidence_file(
        self, book: str, confidence_file: UsfmConfidenceFile, slope: float, intercept: float
    ) -> None:
        confidence = confidence_file.get_book_confidence(book)
        if confidence is not None:
            projected_chrf3 = slope * confidence + intercept
            self.add_score(book, Score(confidence, projected_chrf3))


@dataclass
class SequenceScore(Score):
    sequence_num: int
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
    seen_files: Set[Path] = field(default_factory=set)

    def add_score(self, trg_draft_file_stem: str, score: Score) -> None:
        self.scores[trg_draft_file_stem] = score

    def get_score(self, trg_draft_file_stem: str) -> Optional[Score]:
        return self.scores.get(trg_draft_file_stem)

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


def estimate_quality(linregress_path: Path, confidence_file_paths: List[Path]) -> None:
    linear_regression_result, confidence_files = validate_inputs(linregress_path, confidence_file_paths)
    verse_scores, chapter_scores, book_scores, sequence_scores, txt_file_scores = project_chrf3(
        linear_regression_result, confidence_files
    )
    compute_quality_labels(
        verse_scores,
        chapter_scores,
        book_scores,
        sequence_scores,
        txt_file_scores,
        confidence_files[0].get_path().parent,
    )


def validate_inputs(
    linregress_path: Path, confidence_file_paths: List[Path]
) -> Tuple[LinearRegressionResult, List[ConfidenceFile]]:
    if not linregress_path.exists():
        raise FileNotFoundError(f"Linear regression file {linregress_path} does not exist.")
    elif linregress_path.is_dir():
        pattern = f"{LINREGRESS_PREFIX}.*.json"
        LOGGER.info(f"Searching for files matching {pattern} in directory {linregress_path}.")
        linregress_files = list(linregress_path.glob(pattern))
        if not linregress_files:
            raise ValueError(f"No file matching {pattern} found in directory {linregress_path}.")
        linregress_path = linregress_files[0]
        LOGGER.info(f"Using linear regression file {linregress_path}.")

    if len(confidence_file_paths) == 0:
        raise ValueError("At least one confidence file must be provided.")
    if not all(cf.is_file() for cf in confidence_file_paths):
        missing_files = [str(cf) for cf in confidence_file_paths if not cf.is_file()]
        raise FileNotFoundError(f"The following confidence files do not exist: {', '.join(missing_files)}")

    with open(linregress_path, "r", encoding="utf-8") as f:
        linear_regression_result = LinearRegressionResult.fromJSON(f.read())

    confidence_files: List[ConfidenceFile] = []
    for cf in confidence_file_paths:
        confidence_files.append(ConfidenceFile.from_confidence_file_path(cf))

    return linear_regression_result, confidence_files


def project_chrf3(
    linear_regression_result: LinearRegressionResult, confidence_files: List[ConfidenceFile]
) -> Tuple[List[VerseScore], ChapterScores, BookScores, List[SequenceScore], TxtFileScores]:
    slope = linear_regression_result.slope
    intercept = linear_regression_result.intercept
    LOGGER.info(f"Linear regression data:\n{linear_regression_result.toJSON()}")

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
            book_scores.add_scores_from_confidence_file(
                file_verse_scores[0].vref.book, confidence_file, slope, intercept
            )
        elif isinstance(confidence_file, TxtConfidenceFile):
            file_sequence_scores = SequenceScore.get_scores_from_confidence_file(confidence_file, slope, intercept)
            if not file_sequence_scores:
                LOGGER.warning(f"No sequence scores found in confidence file {confidence_file.get_path()}. Skipping.")
                continue
            sequence_scores += file_sequence_scores
            txt_file_scores.add_scores_from_confidence_file(confidence_file, slope, intercept)
    return verse_scores, chapter_scores, book_scores, sequence_scores, txt_file_scores


class Thresholds:
    GREEN_THRESHOLD = 53.0
    YELLOW_THRESHOLD = 44.5
    GREEN_LABEL = "Green"
    YELLOW_LABEL = "Yellow"
    RED_LABEL = "Red"

    @classmethod
    def return_label(cls, projected_chrf3: float) -> str:
        if projected_chrf3 >= cls.GREEN_THRESHOLD:
            return cls.GREEN_LABEL
        elif projected_chrf3 >= cls.YELLOW_THRESHOLD:
            return cls.YELLOW_LABEL
        else:
            return cls.RED_LABEL


def compute_quality_labels(
    verse_scores: List[VerseScore],
    chapter_scores: ChapterScores,
    book_scores: BookScores,
    sequence_scores: List[SequenceScore],
    txt_file_scores: TxtFileScores,
    output_dir: Path,
) -> None:
    if verse_scores:
        with open(output_dir / "usability_verses.tsv", "w", encoding="utf-8", newline="\n") as verse_file:
            verse_file.write("Book\tChapter\tVerse\tProjected chrF3\tLabel\n")
            for verse_score in verse_scores:
                vref = verse_score.vref
                if vref.verse_num == 0:
                    continue
                if verse_score.projected_chrf3 is None:
                    LOGGER.warning(f"{vref} does not have a projected chrf3. Skipping.")
                    continue

                label = Thresholds.return_label(verse_score.projected_chrf3)

                verse_file.write(
                    f"{vref.book}\t{vref.chapter_num}\t{vref.verse_num}\t{verse_score.projected_chrf3:.2f}\t{label}\n"
                )
        compute_chapter_labels(chapter_scores, output_dir)
        compute_book_labels(book_scores, output_dir)
    if sequence_scores:
        with open(output_dir / "usability_sequences.tsv", "w", encoding="utf-8", newline="\n") as sequence_file:
            sequence_file.write("Trg Draft File\tSequence Number\tProjected chrF3\tLabel\n")
            for sequence_score in sequence_scores:
                if sequence_score.projected_chrf3 is None:
                    LOGGER.warning(f"Sequence {sequence_score.sequence_num} does not have a projected chrf3. Skipping.")
                    continue

                label = Thresholds.return_label(sequence_score.projected_chrf3)

                sequence_file.write(
                    f"{sequence_score.trg_draft_file_stem}\t{sequence_score.sequence_num}\t"
                    f"{sequence_score.projected_chrf3:.2f}\t{label}\n"
                )
        compute_txt_file_labels(txt_file_scores, output_dir)


def compute_chapter_labels(
    chapter_scores: ChapterScores,
    output_dir: Path,
) -> None:
    with open(output_dir / "usability_chapters.tsv", "w", encoding="utf-8", newline="\n") as chapter_file:
        chapter_file.write("Book\tChapter\tProjected chrF3\tLabel\n")
        for book in sorted(chapter_scores.scores, key=lambda b: CANONICAL_ORDER[b]):
            for chapter in sorted(chapter_scores.scores[book]):
                score = chapter_scores.scores[book][chapter]
                label = Thresholds.return_label(score.projected_chrf3)
                chapter_file.write(f"{book}\t{chapter}\t{score.projected_chrf3:.2f}\t{label}\n")


def compute_book_labels(
    book_scores: BookScores,
    output_dir: Path,
) -> None:
    with open(output_dir / "usability_books.tsv", "w", encoding="utf-8", newline="\n") as book_file:
        book_file.write("Book\tProjected chrF3\tLabel\n")
        for book in sorted(book_scores.scores, key=lambda b: CANONICAL_ORDER[b]):
            score = book_scores.scores[book]
            label = Thresholds.return_label(score.projected_chrf3)
            book_file.write(f"{book}\t{score.projected_chrf3:.2f}\t{label}\n")


def compute_txt_file_labels(
    txt_file_scores: TxtFileScores,
    output_dir: Path,
) -> None:
    with open(output_dir / "usability_txt_files.tsv", "w", encoding="utf-8", newline="\n") as txt_file:
        txt_file.write("Trg Draft File\tProjected chrF3\tLabel\n")
        for trg_draft_file_stem in sorted(txt_file_scores.scores):
            score = txt_file_scores.scores[trg_draft_file_stem]
            label = Thresholds.return_label(score.projected_chrf3)
            txt_file.write(f"{trg_draft_file_stem}\t{score.projected_chrf3:.2f}\t{label}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the quality of drafts created by an NMT model.")
    parser.add_argument(
        "linregress_file",
        type=str,
        help="Path relative to MT/experiments to a linregress file containing the confidence-to-chrF3 line of best "
        + f"fit produced by the test step, e.g., project_folder/exp_folder/{LINREGRESS_PREFIX}.5000.json (or "
        + f"{LINREGRESS_PREFIX}.eng.fra.5000.json for an experiment with multiple language pairs). "
        + f"If a directory is provided instead, the first {LINREGRESS_PREFIX}.*.json match is used.",
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

    environment = SilNlpEnv.create_standard_environment()

    using_files = bool(args.confidence_files)
    using_books = bool(args.books)
    using_auto_detect = not using_files and not using_books

    if using_files and using_books:
        raise ValueError("Specify either confidence_files or --books, not both.")

    if (using_books or using_auto_detect) and args.confidence_dir is None:
        raise ValueError("When using --books or auto-detecting confidence files, --confidence-dir must be specified.")
    confidence_dir = environment.get_mt_exp_dir(args.confidence_dir or "")
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

    estimate_quality(environment.get_mt_exp_dir(args.linregress_file), confidence_file_paths)


if __name__ == "__main__":
    main()
