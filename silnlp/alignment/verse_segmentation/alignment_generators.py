import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, List

from machine.corpora import AlignedWordPair

from ...common.corpus import load_corpus, write_corpus
from ..utils import compute_alignment_scores
from .word_alignments import WordAlignments, WordAlignmentsBuilder

LOGGER = logging.getLogger("silnlp.alignment.verse_segmentation.alignment_generators")


class AlignmentGenerator(ABC):
    @abstractmethod
    def generate(
        self, source_passages: List[str], target_passages: List[str]
    ) -> Generator[WordAlignments, None, None]: ...


class AbstractAlignmentGeneratorFactory(ABC):
    @abstractmethod
    def create(self, target_passage_file: Path) -> AlignmentGenerator: ...


class AlignmentAverager:
    @staticmethod
    def average(run_alignment_lines: List[List[str]]) -> List[str]:
        num_runs = len(run_alignment_lines)
        averaged_lines: List[str] = []
        t0 = time.perf_counter()
        for row_idx in range(len(run_alignment_lines[0])):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for lines in run_alignment_lines:
                for pair in AlignedWordPair.from_string(lines[row_idx]):
                    pair_counts[(pair.source_index, pair.target_index)] += 1
            averaged_pairs = [pair for pair, count in pair_counts.items() if (count / num_runs) > 0.5]
            averaged_pairs.sort()
            averaged_lines.append(" ".join(f"{src}-{trg}" for src, trg in averaged_pairs))
        elapsed = time.perf_counter() - t0
        LOGGER.info(
            "Averaging completed in %s",
            f"{int(elapsed // 60)}m {elapsed % 60:.2f}s" if elapsed >= 60 else f"{elapsed:.2f}s",
        )
        return averaged_lines


class NamedEngineAlignmentGenerator(AlignmentGenerator):
    def __init__(self, alignment_method: str, target_passage_file: Path, num_runs: int = 1):
        self._alignment_method = alignment_method
        self._target_passage_file = target_passage_file
        self._num_runs = num_runs

    def generate(self, source_passages: List[str], target_passages: List[str]) -> Generator[WordAlignments, None, None]:

        with TemporaryDirectory() as td:
            align_path = Path(td, "sym-align.txt")
            src_path = Path(td, "src_align.txt")
            trg_path = Path(td, "trg_align.txt")
            write_corpus(src_path, source_passages)
            write_corpus(trg_path, target_passages)

            if self._num_runs == 1:
                self._run_alignment(0, Path(td))

                for src_passage, trg_passage, alignment_line in zip(
                    source_passages, target_passages, load_corpus(align_path)
                ):
                    current_word_alignments: WordAlignmentsBuilder = WordAlignmentsBuilder(
                        len(src_passage.split()), len(trg_passage.split())
                    )
                    current_word_alignments.add_aligments(AlignedWordPair.from_string(alignment_line))
                    yield current_word_alignments.build()
            else:
                run_alignment_lines: List[List[str]] = []
                all_runs_start = time.perf_counter()
                for run_idx in range(self._num_runs):
                    self._run_alignment(run_idx, Path(td))
                    current_run_lines = list(load_corpus(align_path))
                    run_alignment_lines.append(current_run_lines)
                elapsed = time.perf_counter() - all_runs_start
                LOGGER.info(
                    "All %d alignment runs completed in %s",
                    self._num_runs,
                    f"{int(elapsed // 60)}m {elapsed % 60:.2f}s" if elapsed >= 60 else f"{elapsed:.2f}s",
                )

                self._check_alignment_line_counts(run_alignment_lines, source_passages)
                averaged_alignment_lines = AlignmentAverager.average(run_alignment_lines)

                for src_passage, trg_passage, averaged_line in zip(
                    source_passages, target_passages, averaged_alignment_lines
                ):
                    current_word_alignments: WordAlignmentsBuilder = WordAlignmentsBuilder(
                        len(src_passage.split()), len(trg_passage.split())
                    )
                    current_word_alignments.add_aligments(
                        [] if not averaged_line else AlignedWordPair.from_string(averaged_line)
                    )
                    yield current_word_alignments.build()

    def _run_alignment(self, run_index: int, alignment_directory: Path) -> None:
        t0 = time.perf_counter()
        compute_alignment_scores(
            Path(alignment_directory, "src_align.txt"),
            Path(alignment_directory, "trg_align.txt"),
            self._alignment_method,
            Path(alignment_directory, "sym-align.txt"),
            "grow-diag-final-and",
        )
        elapsed = time.perf_counter() - t0
        LOGGER.info(
            "Alignment run %d completed in %s",
            run_index + 1,
            f"{int(elapsed // 60)}m {elapsed % 60:.2f}s" if elapsed >= 60 else f"{elapsed:.2f}s",
        )

    def _check_alignment_line_counts(self, run_alignment_lines: List[List[str]], source_passages: List[str]) -> None:
        expected_rows = len(source_passages)
        for run_idx, run_lines in enumerate(run_alignment_lines):
            if len(run_lines) != expected_rows:
                raise ValueError(
                    f"Alignment run {run_idx + 1} produced {len(run_lines)} rows; " f"expected {expected_rows}."
                )


class NamedEngineAlignmentGeneratorFactory(AbstractAlignmentGeneratorFactory):
    def __init__(self, alignment_method: str, num_runs: int = 1):
        self._alignment_method = alignment_method
        self._num_runs = num_runs

    def create(self, target_passage_file: Path) -> AlignmentGenerator:
        return NamedEngineAlignmentGenerator(self._alignment_method, target_passage_file, self._num_runs)


class FastAlignAlignmentGeneratorFactory(NamedEngineAlignmentGeneratorFactory):
    def __init__(self):
        super().__init__("fast_align")


class EflomalAlignmentGeneratorFactory(NamedEngineAlignmentGeneratorFactory):
    def __init__(self, num_runs: int = 1):
        super().__init__("eflomal", num_runs=num_runs)


class FastAlignConstrainedEflomalAlignmentGenerator(AlignmentGenerator):
    _MAX_CROSSINGS_FOR_FAST_ALIGN = 15

    def __init__(self, target_passage_file: Path, num_runs: int = 1):
        self._target_passage_file = target_passage_file
        self._num_runs = num_runs

    def generate(self, source_passages: List[str], target_passages: List[str]) -> Generator[WordAlignments, None, None]:

        fast_align_generator = FastAlignAlignmentGeneratorFactory().create(self._target_passage_file)
        eflomal_generator = EflomalAlignmentGeneratorFactory(self._num_runs).create(self._target_passage_file)

        fast_align_alignments = list(fast_align_generator.generate(source_passages, target_passages))
        eflomal_alignments = list(eflomal_generator.generate(source_passages, target_passages))
        for fast_align_row, eflomal_row in zip(fast_align_alignments, eflomal_alignments):
            yield eflomal_row.remove_links_crossing_n(self._MAX_CROSSINGS_FOR_FAST_ALIGN, fast_align_row)


class FastAlignConstrainedEflomalAlignmentGeneratorFactory(AbstractAlignmentGeneratorFactory):
    def __init__(self, num_runs: int = 1):
        self._num_runs = num_runs

    def create(self, target_passage_file: Path) -> AlignmentGenerator:
        return FastAlignConstrainedEflomalAlignmentGenerator(target_passage_file, self._num_runs)
