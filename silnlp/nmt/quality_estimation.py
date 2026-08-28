import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, TextIO, Tuple

from machine.quality_estimation import is_book_confidence_unusually_low
from machine.scripture import ALL_BOOK_IDS, VerseRef

from ..common.environment import SilNlpEnv
from ..common.linear_regression import LinearRegressionResult
from ..common.translator import CONFIDENCE_SUFFIX, ConfidenceFile, TxtConfidenceFile, UsfmConfidenceFile
from .test import LINREGRESS_PREFIX

LOGGER = logging.getLogger((__package__ or "") + ".quality_estimation")
CANONICAL_ORDER = {book: i for i, book in enumerate(ALL_BOOK_IDS)}
NO_LINREGRESS_WARNING = (
    "No linear regression file provided. Projected chrF3 scores and "
    "usability labels cannot be computed; the usability files will report confidence only."
)


def project_chrf3(linear_regression_result: Optional[LinearRegressionResult], confidence: float) -> Optional[float]:
    if linear_regression_result is None:
        return None
    return linear_regression_result.slope * confidence + linear_regression_result.intercept


@dataclass
class Score:
    confidence: float
    projected_chrf3: Optional[float]


@dataclass
class VerseScore(Score):
    vref: VerseRef

    @classmethod
    def get_scores_from_confidence_file(
        cls, confidence_file: UsfmConfidenceFile, linear_regression_result: Optional[LinearRegressionResult]
    ) -> List["VerseScore"]:
        verse_scores: List[VerseScore] = []
        for vref, confidence in confidence_file.verse_confidence_iterator():
            verse_scores.append(cls(confidence, project_chrf3(linear_regression_result, confidence), vref))
        return verse_scores


@dataclass
class ChapterScores:
    scores: Dict[str, Dict[int, Score]] = field(default_factory=lambda: defaultdict(dict))

    def add_score(self, book: str, chapter: int, score: Score) -> None:
        self.scores[book][chapter] = score

    def get_score(self, book: str, chapter: int) -> Optional[Score]:
        return self.scores.get(book, {}).get(chapter)

    def add_scores_from_confidence_file(
        self,
        book: str,
        confidence_file: UsfmConfidenceFile,
        linear_regression_result: Optional[LinearRegressionResult],
    ) -> None:
        for chapter, confidence in confidence_file.chapter_confidence_iterator():
            score = Score(confidence, project_chrf3(linear_regression_result, confidence))
            self.add_score(book, chapter, score)


@dataclass
class BookScores:
    scores: Dict[str, Score] = field(default_factory=dict)

    def add_score(self, book: str, score: Score) -> None:
        self.scores[book] = score

    def get_score(self, book: str) -> Optional[Score]:
        return self.scores.get(book)

    def add_scores_from_confidence_file(
        self,
        book: str,
        confidence_file: UsfmConfidenceFile,
        linear_regression_result: Optional[LinearRegressionResult],
    ) -> None:
        confidence = confidence_file.get_book_confidence(book)
        if confidence is not None:
            self.add_score(book, Score(confidence, project_chrf3(linear_regression_result, confidence)))


@dataclass
class SequenceScore(Score):
    sequence_num: int
    trg_draft_file_stem: str

    @classmethod
    def get_scores_from_confidence_file(
        cls, confidence_file: TxtConfidenceFile, linear_regression_result: Optional[LinearRegressionResult]
    ) -> List["SequenceScore"]:
        trg_draft_file_stem = confidence_file.get_trg_draft_file_path().stem
        sequence_scores: List[SequenceScore] = []
        for sequence_num, confidence in confidence_file.verse_confidence_iterator():
            sequence_scores.append(
                cls(
                    confidence,
                    project_chrf3(linear_regression_result, confidence),
                    sequence_num,
                    trg_draft_file_stem,
                )
            )
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
        self, confidence_file: TxtConfidenceFile, linear_regression_result: Optional[LinearRegressionResult]
    ) -> None:
        files_path = confidence_file.get_files_path()
        if files_path.is_file() and files_path not in self.seen_files:
            self.seen_files.add(files_path)
            for trg_draft_file_stem, confidence in confidence_file.file_confidence_iterator():
                score = Score(confidence, project_chrf3(linear_regression_result, confidence))
                self.add_score(trg_draft_file_stem, score)


def estimate_quality(linregress_path: Optional[Path], confidence_file_paths: List[Path]) -> None:
    linear_regression_result, confidence_files = validate_inputs(linregress_path, confidence_file_paths)
    verse_scores, chapter_scores, book_scores, sequence_scores, txt_file_scores = compute_scores(
        linear_regression_result, confidence_files
    )
    compute_quality_labels(
        verse_scores,
        chapter_scores,
        book_scores,
        sequence_scores,
        txt_file_scores,
        confidence_files[0].get_path().parent,
        linear_regression_result is not None,
    )


def validate_inputs(
    linregress_path: Optional[Path], confidence_file_paths: List[Path]
) -> Tuple[Optional[LinearRegressionResult], List[ConfidenceFile]]:
    if linregress_path is None:
        LOGGER.warning(NO_LINREGRESS_WARNING)
    elif not linregress_path.exists():
        raise FileNotFoundError(f"Linear regression file {linregress_path} does not exist.")
    elif linregress_path.is_dir():
        pattern = f"{LINREGRESS_PREFIX}.*.json"
        LOGGER.info(f"Searching for files matching {pattern} in directory {linregress_path}.")
        linregress_files = list(linregress_path.glob(pattern))
        if not linregress_files:
            LOGGER.warning(f"No file matching {pattern} found in directory {linregress_path}. {NO_LINREGRESS_WARNING}")
            linregress_path = None
        else:
            linregress_path = linregress_files[0]
            LOGGER.info(f"Using linear regression file {linregress_path}.")

    if len(confidence_file_paths) == 0:
        raise ValueError("At least one confidence file must be provided.")
    if not all(cf.is_file() for cf in confidence_file_paths):
        missing_files = [str(cf) for cf in confidence_file_paths if not cf.is_file()]
        raise FileNotFoundError(f"The following confidence files do not exist: {', '.join(missing_files)}")

    linear_regression_result: Optional[LinearRegressionResult] = None
    if linregress_path is not None:
        with open(linregress_path, "r", encoding="utf-8") as f:
            linear_regression_result = LinearRegressionResult.fromJSON(f.read())

    confidence_files: List[ConfidenceFile] = []
    for cf in confidence_file_paths:
        confidence_files.append(ConfidenceFile.from_confidence_file_path(cf))

    return linear_regression_result, confidence_files


def compute_scores(
    linear_regression_result: Optional[LinearRegressionResult], confidence_files: List[ConfidenceFile]
) -> Tuple[List[VerseScore], ChapterScores, BookScores, List[SequenceScore], TxtFileScores]:
    if linear_regression_result is not None:
        LOGGER.info(f"Linear regression data:\n{linear_regression_result.toJSON()}")

    verse_scores: List[VerseScore] = []
    chapter_scores: ChapterScores = ChapterScores()
    book_scores: BookScores = BookScores()
    sequence_scores: List[SequenceScore] = []
    txt_file_scores: TxtFileScores = TxtFileScores()
    for confidence_file in confidence_files:
        if isinstance(confidence_file, UsfmConfidenceFile):
            file_verse_scores = VerseScore.get_scores_from_confidence_file(confidence_file, linear_regression_result)
            if not file_verse_scores:
                LOGGER.warning(f"No verse scores found in confidence file {confidence_file.get_path()}. Skipping.")
                continue
            verse_scores += file_verse_scores
            chapter_scores.add_scores_from_confidence_file(
                file_verse_scores[0].vref.book, confidence_file, linear_regression_result
            )
            book_scores.add_scores_from_confidence_file(
                file_verse_scores[0].vref.book, confidence_file, linear_regression_result
            )
        elif isinstance(confidence_file, TxtConfidenceFile):
            file_sequence_scores = SequenceScore.get_scores_from_confidence_file(
                confidence_file, linear_regression_result
            )
            if not file_sequence_scores:
                LOGGER.warning(f"No sequence scores found in confidence file {confidence_file.get_path()}. Skipping.")
                continue
            sequence_scores += file_sequence_scores
            txt_file_scores.add_scores_from_confidence_file(confidence_file, linear_regression_result)
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


def chrf3_headers(include_projected_chrf3: bool) -> List[str]:
    return ["Projected chrF3", "Label"] if include_projected_chrf3 else []


def chrf3_cells(projected_chrf3: Optional[float], include_projected_chrf3: bool) -> List[str]:
    if not include_projected_chrf3:
        return []
    if projected_chrf3 is None:
        return ["", ""]
    return [f"{projected_chrf3:.2f}", Thresholds.return_label(projected_chrf3)]


def write_row(file: TextIO, cells: List[str]) -> None:
    file.write("\t".join(cells) + "\n")


def compute_quality_labels(
    verse_scores: List[VerseScore],
    chapter_scores: ChapterScores,
    book_scores: BookScores,
    sequence_scores: List[SequenceScore],
    txt_file_scores: TxtFileScores,
    output_dir: Path,
    include_projected_chrf3: bool,
) -> None:
    if verse_scores:
        with open(output_dir / "usability_verses.tsv", "w", encoding="utf-8", newline="\n") as verse_file:
            write_row(verse_file, ["Book", "Chapter", "Verse", "Confidence"] + chrf3_headers(include_projected_chrf3))
            for verse_score in verse_scores:
                vref = verse_score.vref
                if vref.verse_num == 0:
                    continue
                if include_projected_chrf3 and verse_score.projected_chrf3 is None:
                    LOGGER.warning(f"{vref} does not have a projected chrf3. Skipping.")
                    continue

                write_row(
                    verse_file,
                    [str(vref.book), str(vref.chapter_num), str(vref.verse_num), f"{verse_score.confidence:.4f}"]
                    + chrf3_cells(verse_score.projected_chrf3, include_projected_chrf3),
                )
        compute_chapter_labels(chapter_scores, output_dir, include_projected_chrf3)
        compute_book_labels(book_scores, output_dir, include_projected_chrf3)
    if sequence_scores:
        with open(output_dir / "usability_sequences.tsv", "w", encoding="utf-8", newline="\n") as sequence_file:
            write_row(
                sequence_file,
                ["Trg Draft File", "Sequence Number", "Confidence"] + chrf3_headers(include_projected_chrf3),
            )
            for sequence_score in sequence_scores:
                if include_projected_chrf3 and sequence_score.projected_chrf3 is None:
                    LOGGER.warning(f"Sequence {sequence_score.sequence_num} does not have a projected chrf3. Skipping.")
                    continue

                write_row(
                    sequence_file,
                    [
                        sequence_score.trg_draft_file_stem,
                        str(sequence_score.sequence_num),
                        f"{sequence_score.confidence:.4f}",
                    ]
                    + chrf3_cells(sequence_score.projected_chrf3, include_projected_chrf3),
                )
        compute_txt_file_labels(txt_file_scores, output_dir, include_projected_chrf3)


def compute_chapter_labels(
    chapter_scores: ChapterScores,
    output_dir: Path,
    include_projected_chrf3: bool,
) -> None:
    with open(output_dir / "usability_chapters.tsv", "w", encoding="utf-8", newline="\n") as chapter_file:
        write_row(chapter_file, ["Book", "Chapter", "Confidence"] + chrf3_headers(include_projected_chrf3))
        for book in sorted(chapter_scores.scores, key=lambda b: CANONICAL_ORDER[b]):
            for chapter in sorted(chapter_scores.scores[book]):
                score = chapter_scores.scores[book][chapter]
                write_row(
                    chapter_file,
                    [book, str(chapter), f"{score.confidence:.4f}"]
                    + chrf3_cells(score.projected_chrf3, include_projected_chrf3),
                )


def compute_book_labels(
    book_scores: BookScores,
    output_dir: Path,
    include_projected_chrf3: bool,
) -> None:
    with open(output_dir / "usability_books.tsv", "w", encoding="utf-8", newline="\n") as book_file:
        write_row(book_file, ["Book", "Confidence", "Low Confidence"] + chrf3_headers(include_projected_chrf3))
        for book in sorted(book_scores.scores, key=lambda b: CANONICAL_ORDER[b]):
            score = book_scores.scores[book]
            low_confidence = is_book_confidence_unusually_low(score.confidence, book_id=book)
            write_row(
                book_file,
                [book, f"{score.confidence:.4f}", str(low_confidence)]
                + chrf3_cells(score.projected_chrf3, include_projected_chrf3),
            )


def compute_txt_file_labels(
    txt_file_scores: TxtFileScores,
    output_dir: Path,
    include_projected_chrf3: bool,
) -> None:
    with open(output_dir / "usability_txt_files.tsv", "w", encoding="utf-8", newline="\n") as txt_file:
        write_row(txt_file, ["Trg Draft File", "Confidence"] + chrf3_headers(include_projected_chrf3))
        for trg_draft_file_stem in sorted(txt_file_scores.scores):
            score = txt_file_scores.scores[trg_draft_file_stem]
            write_row(
                txt_file,
                [trg_draft_file_stem, f"{score.confidence:.4f}"]
                + chrf3_cells(score.projected_chrf3, include_projected_chrf3),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the quality of drafts created by an NMT model.")
    parser.add_argument(
        "linregress_file",
        nargs="?",
        default=None,
        type=str,
        help="Path relative to MT/experiments to a linregress file containing the confidence-to-chrF3 line of best "
        + f"fit produced by the test step, e.g., project_folder/exp_folder/{LINREGRESS_PREFIX}.5000.json (or "
        + f"{LINREGRESS_PREFIX}.eng.fra.5000.json for an experiment with multiple language pairs). "
        + f"If a directory is provided instead, the first {LINREGRESS_PREFIX}.*.json match is used. "
        + "Omit this argument (and use --confidence-dir or --experiment-dir) to run without a test set, in which "
        + "case the usability files report confidence only.",
    )
    parser.add_argument(
        "confidence_files",
        nargs="*",
        type=str,
        help="Zero or more confidence file paths (.confidences.tsv suffix, e.g., "
        + "project_folder/exp_folder/infer/5000/source/631JN.SFM.confidences.tsv'). Paths are relative to "
        + "MT/experiments by default or to MT/experiments/--confidence-dir if --confidence-dir is specified. "
        + "If zero paths are provided, either --confidence-dir or --experiment-dir must be used to autodetect files.",
    )
    parser.add_argument(
        "--confidence-dir",
        type=str,
        default=None,
        help="Directory relative to MT/experiments containing confidence files.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Directory relative to MT/experiments to an experiment folder. Quality estimation is run "
        + "for every directory under <experiment-dir>/infer that contains confidence files.",
    )
    args = parser.parse_args()

    if args.linregress_file is not None and args.linregress_file.endswith(CONFIDENCE_SUFFIX):
        args.confidence_files.insert(0, args.linregress_file)
        args.linregress_file = None

    environment = SilNlpEnv.create_standard_environment()
    linregress_path = environment.get_mt_exp_dir(args.linregress_file) if args.linregress_file is not None else None

    using_experiment_dir = args.experiment_dir is not None
    using_files = bool(args.confidence_files)

    if using_experiment_dir and (using_files or args.confidence_dir is not None):
        raise ValueError(
            "--experiment-dir cannot be combined with confidence_files positional arg or --confidence-dir."
        )

    confidence_files_per_directory: List[List[Path]]
    if using_experiment_dir:
        infer_dir = environment.get_mt_exp_dir(args.experiment_dir) / "infer"
        if not infer_dir.is_dir():
            raise ValueError(f"Infer directory {infer_dir} does not exist or is not a directory.")
        confidence_dirs = sorted({cf.parent for cf in infer_dir.rglob(f"*{CONFIDENCE_SUFFIX}")})
        if not confidence_dirs:
            raise ValueError(f"No confidence files found under {infer_dir}.")
        num_dirs = len(confidence_dirs)
        LOGGER.info(
            f"Auto-detecting confidence files in {num_dirs} director{'y' if num_dirs == 1 else 'ies'} "
            + f"under {infer_dir}."
        )
        confidence_files_per_directory = [sorted(d.glob(f"*{CONFIDENCE_SUFFIX}")) for d in confidence_dirs]
    else:
        if not using_files and args.confidence_dir is None:
            raise ValueError(
                "Did not provide one of these args: confidence_files, --confidence-dir, or --experiment-dir."
            )
        confidence_dir = environment.get_mt_exp_dir(args.confidence_dir or "")
        if not confidence_dir.is_dir():
            raise ValueError(f"Confidence directory {confidence_dir} does not exist or is not a directory.")

        if using_files:
            confidence_file_paths = [confidence_dir / confidence_file for confidence_file in args.confidence_files]
        else:
            LOGGER.info(f"Auto-detecting confidence files in directory {confidence_dir}")
            confidence_file_paths = list(confidence_dir.glob(f"*{CONFIDENCE_SUFFIX}"))
            if not confidence_file_paths:
                raise ValueError(f"No confidence files found in {confidence_dir}.")
        confidence_files_per_directory = [confidence_file_paths]

    for confidence_file_paths in confidence_files_per_directory:
        try:
            estimate_quality(linregress_path, confidence_file_paths)
        except Exception:
            LOGGER.exception(f"Quality estimation failed for {confidence_file_paths[0].parent}.")


if __name__ == "__main__":
    main()
