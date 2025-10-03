import argparse
import random
import shutil
from abc import ABC, abstractmethod
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from machine.corpora import FileParatextProjectSettingsParser, ScriptureRef, UsfmFileText
from machine.scripture import VerseRef
from machine.tokenization import LatinWordTokenizer
from machine.translation import WordAlignmentMatrix
from pyparsing import Iterable

from silnlp.common.environment import SIL_NLP_ENV

from ..common.corpus import load_corpus, write_corpus
from .eflomal import to_word_alignment_matrix
from .utils import compute_alignment_scores


@dataclass
class Passage:
    start_ref: VerseRef
    end_ref: VerseRef
    text: str

    def is_ref_in_range(self, ref: VerseRef) -> bool:
        return ref.compare_to(self.start_ref) >= 0 and ref.compare_to(self.end_ref) <= 0


@dataclass
class Verse:
    reference: VerseRef
    text: str


@dataclass
class SegmentedPassage:
    start_ref: VerseRef
    end_ref: VerseRef
    verses: List[Verse]

    def is_ref_in_range(self, ref: VerseRef) -> bool:
        return ref.compare_to(self.start_ref) >= 0 and ref.compare_to(self.end_ref) <= 0


class PassageReader:
    def __init__(self, passage_file: Path):
        self._passages = self._read_passage_file(passage_file)

    def _read_passage_file(self, passage_file: Path) -> List[Passage]:
        passages = []
        for line in load_corpus(passage_file):
            row = line.split("\t")
            if len(row) < 7:
                continue  # skip malformed lines
            passage = Passage(
                VerseRef(row[0], int(row[1]), int(row[2])),
                VerseRef(row[3], int(row[4]), int(row[5])),
                text=row[6],
            )
            passages.append(passage)
        return passages

    def get_passages(self) -> List[Passage]:
        return self._passages


class ParallelPassage:
    PROHIBITED_VERSE_STARTING_CHARACTERS: Set[str] = {
        " ",
        ",",
        ";",
        ":",
        ".",
        "!",
        "?",
        ")",
        "]",
        "}",
        "»",
        "›",
        "”",
        "’",
    }
    PROHIBITED_VERSE_ENDING_CHARACTERS: Set[str] = {"(", "[", "{", "«", "‹", "“", "‘"}

    def __init__(
        self,
        start_ref: VerseRef,
        end_ref: VerseRef,
        source_verses: List[Verse],
        target_text: str,
        word_alignment_matrix: Optional[WordAlignmentMatrix] = None,
    ):
        self._start_ref = start_ref
        self._end_ref = end_ref
        self._source_verses = source_verses
        self._target_text = target_text
        self._word_alignment_matrix = word_alignment_matrix
        self._cached_crossed_alignments: Dict[tuple[int, int], int] = {}

    def tokenize_text(self) -> None:
        tokenizer = LatinWordTokenizer()
        self._tokenized_source_verses = [list(tokenizer.tokenize(verse.text)) for verse in self._source_verses]
        self._tokenized_source_text = [token for verse in self._tokenized_source_verses for token in verse]
        self._source_verse_token_offsets = self._calculate_verse_token_offsets()
        self._tokenized_target_text = list(tokenizer.tokenize(self._target_text))

    def _calculate_verse_token_offsets(self) -> List[int]:
        offsets = []
        current_offset = 0
        for verse in self._tokenized_source_verses:
            current_offset += len(verse)
            offsets.append(current_offset)
        return offsets

    def is_ref_in_range(self, ref: VerseRef) -> bool:
        return ref.compare_to(self._start_ref) >= 0 and ref.compare_to(self._end_ref) <= 0

    def add_source_verse(self, source_verse: Verse) -> None:
        self._source_verses.append(source_verse)

    def get_source_text(self) -> str:
        return " ".join([verse.text for verse in self._source_verses])

    def get_token_separated_source_text(self) -> str:
        tokenizer = LatinWordTokenizer()
        tokens = []
        for verse in self._source_verses:
            tokens.extend([token for token in tokenizer.tokenize(verse.text)])
        return " ".join(tokens)

    def get_target_text(self) -> str:
        return self._target_text

    def get_token_separated_target_text(self) -> str:
        tokenizer = LatinWordTokenizer()
        tokens = [token for token in tokenizer.tokenize(self._target_text)]
        return " ".join(tokens)

    def load_word_alignment_matrix(self, alignment_line: str) -> None:
        self._word_alignment_matrix = to_word_alignment_matrix(alignment_line)
        self._word_alignments = self._word_alignment_matrix.to_aligned_word_pairs()

    # TODO: remove
    def _calculate_crossed_alignments(self) -> npt.NDArray[np.int_]:
        crossed_alignments = np.zeros(
            shape=(self._word_alignment_matrix.row_count, self._word_alignment_matrix.column_count), dtype=np.int_
        )
        for src_word_index in range(self._word_alignment_matrix.row_count):
            for aligned_trg_word_index in self._word_alignment_matrix.get_row_aligned_indices(src_word_index):
                for earlier_src_index in range(src_word_index):
                    for later_trg_index in range(aligned_trg_word_index + 1, self._word_alignment_matrix.column_count):
                        crossed_alignments[earlier_src_index, later_trg_index] += 1
                for later_src_index in range(src_word_index + 1, self._word_alignment_matrix.row_count):
                    for earlier_trg_index in range(0, aligned_trg_word_index):
                        crossed_alignments[later_src_index, earlier_trg_index] += 1
        return crossed_alignments

    def _get_num_crossed_alignments(self, src_word_index: int, trg_word_index: int) -> int:
        if (src_word_index, trg_word_index) in self._cached_crossed_alignments:
            return self._cached_crossed_alignments[(src_word_index, trg_word_index)]
        num_crossings = 0
        for aligned_word_pair in self._word_alignments:
            if (
                aligned_word_pair.source_index < src_word_index and aligned_word_pair.target_index > trg_word_index
            ) or (aligned_word_pair.source_index > src_word_index and aligned_word_pair.target_index < trg_word_index):
                num_crossings += 1
        self._cached_crossed_alignments[(src_word_index, trg_word_index)] = num_crossings
        return num_crossings

    def segment_target_passage(self) -> SegmentedPassage:
        last_target_verse_offset = -1
        target_verse_offsets: List[int] = []
        for verse_index, (source_verse, verse_token_offset) in enumerate(
            zip(self._source_verses[:-1], self._source_verse_token_offsets[:-1])
        ):
            fewest_crossed_alignments = 1000000
            best_split_indices = []
            for trg_word_index in range(last_target_verse_offset + 1, self._word_alignment_matrix.column_count):
                crossed_alignments = self._get_num_crossed_alignments(verse_token_offset - 1, trg_word_index)

                if trg_word_index < self._word_alignment_matrix.column_count - 1:
                    crossed_alignments += self._get_num_crossed_alignments(verse_token_offset, trg_word_index)

                if crossed_alignments == fewest_crossed_alignments:
                    best_split_indices.append(trg_word_index)
                elif crossed_alignments < fewest_crossed_alignments:
                    fewest_crossed_alignments = crossed_alignments
                    best_split_indices = [trg_word_index]
            if len(best_split_indices) == 0:
                target_verse_offsets.append(-1)
            else:
                target_verse_offsets.append(random.choice(best_split_indices))
            last_target_verse_offset = target_verse_offsets[-1]
        return self._create_target_verses_from_offsets(target_verse_offsets)

    def _create_target_verses_from_offsets(self, target_verse_offsets: List[int]) -> SegmentedPassage:
        target_verses = []

        current_verse_starting_char_index = 0
        current_verse_ending_char_index = 0
        current_verse_offset_index = 0
        for target_word_index, target_word in enumerate(self._tokenized_target_text):
            if target_verse_offsets[current_verse_offset_index] == -1:
                verse_ref = self._source_verses[current_verse_offset_index].reference
                target_verses.append(Verse(verse_ref, ""))

                current_verse_starting_char_index = current_verse_ending_char_index
                current_verse_offset_index += 1
                if current_verse_offset_index >= len(target_verse_offsets):
                    break

            elif target_word_index == target_verse_offsets[current_verse_offset_index]:
                verse_ref = self._source_verses[current_verse_offset_index].reference
                verse_text = self._target_text[current_verse_starting_char_index:current_verse_ending_char_index]
                target_verses.append(Verse(verse_ref, verse_text))

                current_verse_starting_char_index = current_verse_ending_char_index
                current_verse_offset_index += 1
                if current_verse_offset_index >= len(target_verse_offsets):
                    break

            current_verse_ending_char_index = self._target_text.index(
                target_word, current_verse_ending_char_index
            ) + len(target_word)

        last_verse_ref = self._source_verses[-1].reference
        last_verse_text = self._target_text[current_verse_starting_char_index:]
        target_verses.append(Verse(last_verse_ref, last_verse_text))

        return SegmentedPassage(self._start_ref, self._end_ref, self._adjust_verse_boundaries(target_verses))

    def _adjust_verse_boundaries(self, target_verses: List[Verse]) -> List[Verse]:
        for verse, next_verse in zip(target_verses[:-1], target_verses[1:]):
            while len(next_verse.text) > 0 and next_verse.text[0] in self.PROHIBITED_VERSE_STARTING_CHARACTERS:
                verse.text += next_verse.text[0]
                next_verse.text = next_verse.text[1:]
            while len(verse.text) > 0 and verse.text[-1] in self.PROHIBITED_VERSE_ENDING_CHARACTERS:
                next_verse.text = verse.text[-1] + next_verse.text
                verse.text = verse.text[:-1]
        return target_verses


class ParallelPassageCollectionFactory:
    def __init__(self, save_alignments: bool = False, use_saved_alignments: bool = False):
        self._save_alignments = save_alignments
        self._use_saved_alignments = use_saved_alignments

    def create(self, source_project_name: str, target_passage_file: Path) -> "ParallelPassageCollection":
        if self._use_saved_alignments:
            alignment_file = target_passage_file.with_suffix(".alignments.txt")
            if not alignment_file.exists():
                raise FileNotFoundError(f"Saved alignment file {alignment_file} not found")
            saved_alignment_generator = SavedAlignmentGenerator(alignment_file)
            return ParallelPassageCollection(source_project_name, target_passage_file, saved_alignment_generator)
        return ParallelPassageCollection(
            source_project_name,
            target_passage_file,
            EflomalAlignmentGenerator(self._save_alignments, target_passage_file),
        )


class AlignmentGenerator(ABC):
    @abstractmethod
    def generate(self, source_passages: List[str], target_passages: List[str]) -> List[str]:
        pass


class EflomalAlignmentGenerator(AlignmentGenerator):
    def __init__(self, save_alignments: bool = False, target_passage_file: Optional[Path] = None):
        self._save_alignments = save_alignments
        self._target_passage_file = target_passage_file

    def generate(self, source_passages: List[str], target_passages: List[str]) -> List[str]:
        with TemporaryDirectory() as td:
            align_path = Path(td, "sym-align.txt")
            src_path = Path(td, "src_align.txt")
            trg_path = Path(td, "trg_align.txt")
            write_corpus(src_path, source_passages)
            write_corpus(trg_path, target_passages)
            compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), "eflomal", align_path)

            for line in load_corpus(align_path):
                yield line

            if self._save_alignments:
                assert self._target_passage_file is not None
                saved_alignments_file = self._target_passage_file.with_suffix(".alignments.txt")
                shutil.copy(align_path, saved_alignments_file)


class SavedAlignmentGenerator(AlignmentGenerator):
    def __init__(self, alignment_file: Path):
        self._alignment_file = alignment_file

    def generate(self, source_passages: List[str], target_passages: List[str]) -> List[str]:
        return load_corpus(self._alignment_file)


class ParallelPassageCollection:
    def __init__(self, source_project_name: str, target_passage_file: Path, alignment_generator: AlignmentGenerator):
        target_passages = PassageReader(target_passage_file).get_passages()
        self._collect_parallel_passages(source_project_name, target_passages)
        self._create_word_alignment_matrix(alignment_generator)

    def _get_all_books_in_passages(self, passages: List[Passage]) -> Set[str]:
        all_books: Set[str] = set()
        for passage in passages:
            all_books.add(passage.start_ref.book)
            all_books.add(passage.end_ref.book)
        return all_books

    def _collect_parallel_passages(self, source_project_name: str, target_passages: List[Passage]) -> None:
        self._parallel_passages = [ParallelPassage(t.start_ref, t.end_ref, [], t.text) for t in target_passages]

        settings = FileParatextProjectSettingsParser(SIL_NLP_ENV.pt_projects_dir / source_project_name).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        for book in self._get_all_books_in_passages(target_passages):
            usfm_text = UsfmFileText(
                stylesheet,
                encoding,
                book,
                SIL_NLP_ENV.pt_projects_dir / source_project_name / settings.get_book_file_name(book),
            )
            for row in usfm_text:
                for parallel_passage in self._parallel_passages:
                    if isinstance(row.ref, ScriptureRef) and parallel_passage.is_ref_in_range(row.ref.verse_ref):
                        parallel_passage.add_source_verse(Verse(row.ref.verse_ref, row.text))
        for passage in self._parallel_passages:
            passage.tokenize_text()

    def _create_word_alignment_matrix(self, alignment_generator: AlignmentGenerator) -> None:
        for index, alignment_line in enumerate(
            alignment_generator.generate(
                [passage.get_token_separated_source_text() for passage in self._parallel_passages],
                [passage.get_token_separated_target_text() for passage in self._parallel_passages],
            )
        ):
            self._parallel_passages[index].load_word_alignment_matrix(alignment_line)

    def segment_target_passages(self) -> Iterable[SegmentedPassage]:
        for passage in self._parallel_passages:
            yield passage.segment_target_passage()


class ReferenceVerseSegmentationReader:
    def read_passages(self, target_project_name: str, target_passage_file: Path):
        self._passages = PassageReader(target_passage_file).get_passages()
        return self._read_segmented_passages(target_project_name)

    def _read_segmented_passages(self, target_project_name: str) -> List[SegmentedPassage]:
        segmented_passages: List[SegmentedPassage] = [
            SegmentedPassage(p.start_ref, p.end_ref, []) for p in self._passages
        ]

        settings = FileParatextProjectSettingsParser(SIL_NLP_ENV.pt_projects_dir / target_project_name).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        for segmented_passage in segmented_passages:
            usfm_text = UsfmFileText(
                stylesheet,
                encoding,
                segmented_passage.start_ref.book,
                SIL_NLP_ENV.pt_projects_dir
                / target_project_name
                / settings.get_book_file_name(segmented_passage.start_ref.book),
            )
            for row in usfm_text:
                if isinstance(row.ref, ScriptureRef) and segmented_passage.is_ref_in_range(row.ref.verse_ref):
                    segmented_passage.verses.append(Verse(row.ref.verse_ref, row.text))
        return segmented_passages


class SegmentationEvaluation:
    def __init__(self):
        self._num_breaks = 0
        self._num_correct_breaks = 0
        self._total_distance = 0

    def record_verse_break(self, distance_from_reference: int) -> None:
        self._num_breaks += 1
        if distance_from_reference == 0:
            self._num_correct_breaks += 1
        self._total_distance += distance_from_reference

    def get_accuracy(self) -> float:
        return self._num_correct_breaks / self._num_breaks if self._num_breaks > 0 else 0.0

    def get_average_distance(self) -> float:
        return self._total_distance / self._num_breaks if self._num_breaks > 0 else 0.0


class SegmentationEvaluator:
    def __init__(self, reference_segmentations: List[SegmentedPassage]):
        self._reference_segmentations = reference_segmentations

    def evaluate_segmentation(self, segmented_passages: List[SegmentedPassage]) -> SegmentationEvaluation:
        if len(self._reference_segmentations) != len(segmented_passages):
            raise ValueError("Number of segmented passages does not match number of reference segmentations")

        segmentation_evaluation = SegmentationEvaluation()
        for reference_passage, segmented_passage in zip(self._reference_segmentations, segmented_passages):
            if (
                reference_passage.start_ref != segmented_passage.start_ref
                or reference_passage.end_ref != segmented_passage.end_ref
            ):
                raise ValueError(
                    f"Passage references do not match: {reference_passage.start_ref}-{reference_passage.end_ref} vs {segmented_passage.start_ref}-{segmented_passage.end_ref}"
                )
            self._check_passage_lengths(reference_passage, segmented_passage)
            self._evaluate_single_passage(reference_passage, segmented_passage, segmentation_evaluation)
        return segmentation_evaluation

    def _check_passage_lengths(self, reference_passage: SegmentedPassage, segmented_passage: SegmentedPassage) -> None:
        reference_length = sum(len(self._normalize_verse_text(verse.text)) for verse in reference_passage.verses)
        segmented_length = sum(
            len(self._normalize_verse_text(verse.text)) for verse in segmented_passage.verses if len(verse.text) > 0
        )
        if reference_length != segmented_length:
            raise ValueError(
                f"Passage lengths do not match for {reference_passage.start_ref}-{reference_passage.end_ref}: {reference_length} vs {segmented_length}"
            )

    def _normalize_verse_text(self, text: str) -> str:
        return text.replace(" ", "")

    def _evaluate_single_passage(
        self,
        reference_passage: SegmentedPassage,
        segmented_passage: SegmentedPassage,
        segmentation_evaluation: SegmentationEvaluation,
    ) -> None:
        reference_running_character_offset = 0
        segmented_running_character_offset = 0
        for reference_verse, segmented_verse in zip(reference_passage.verses[:-1], segmented_passage.verses[:-1]):
            reference_running_character_offset += len(self._normalize_verse_text(reference_verse.text))
            segmented_running_character_offset += len(self._normalize_verse_text(segmented_verse.text))
            distance = abs(reference_running_character_offset - segmented_running_character_offset)
            segmentation_evaluation.record_verse_break(distance)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect verse counts and compute alignment scores")
    parser.add_argument("--source-project", help="Name of source Paratext project", required=True, type=str)
    parser.add_argument("--target-passages", help=".tsv file with target passages", required=True, type=str)
    parser.add_argument(
        "--compare-against", help="Name of Paratext project where target passages come from", default=None, type=str
    )
    parser.add_argument(
        "--save-alignments", help="Save the computed alignments for future use", default=None, action="store_true"
    )
    parser.add_argument(
        "--use-saved-alignments", help="Save the computed alignments for future use", default=None, action="store_true"
    )
    args = parser.parse_args()

    parallel_passages = ParallelPassageCollectionFactory(args.save_alignments, args.use_saved_alignments).create(
        args.source_project, Path(args.target_passages)
    )
    segmented_passages = list(parallel_passages.segment_target_passages())
    for segmented_passage in segmented_passages:
        for verse in segmented_passage.verses:
            print(f"{verse.reference}\t{verse.text}")
    if args.compare_against is not None:
        reference_segmentations = ReferenceVerseSegmentationReader().read_passages(
            args.compare_against, Path(args.target_passages)
        )
        segmentation_evaluator = SegmentationEvaluator(reference_segmentations)
        segmentation_evaluation = segmentation_evaluator.evaluate_segmentation(segmented_passages)
        print(
            f"Segmentation accuracy = {segmentation_evaluation.get_accuracy() * 100}%, average distance = {segmentation_evaluation.get_average_distance()} characters"
        )


if __name__ == "__main__":
    main()
