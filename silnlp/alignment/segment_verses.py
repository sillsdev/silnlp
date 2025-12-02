import argparse
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Collection, Dict, Generator, List, Optional, TextIO

import regex
from machine.corpora import AlignedWordPair, FileParatextProjectSettingsParser, ScriptureRef, UsfmFileText
from machine.scripture import VerseRef
from machine.tokenization import LatinWordTokenizer
from pyparsing import Iterable

from silnlp.common.environment import SIL_NLP_ENV

from ..common.corpus import load_corpus, write_corpus
from .utils import compute_alignment_scores

unicode_letter_regex = regex.compile("\\p{L}")


def contains_letter(token: str) -> bool:
    return unicode_letter_regex.search(token)


@dataclass
class VerseRange:
    start_ref: VerseRef
    end_ref: VerseRef

    def is_ref_in_range(self, ref: VerseRef) -> bool:
        return ref.compare_to(self.start_ref) >= 0 and ref.compare_to(self.end_ref) <= 0


@dataclass
class Verse:
    reference: VerseRef
    text: str


class VerseCollector(VerseRange):
    @abstractmethod
    def add_verse(self, verse: Verse) -> None: ...


@dataclass
class Passage(VerseRange):
    text: str


@dataclass
class SegmentedPassage(Passage):
    verses: List[Verse]

    def write_to_file(self, file: TextIO) -> None:
        for verse in self.verses:
            file.write(verse.text + "\n")


class SegmentedPassageBuilder(VerseCollector):
    def __init__(self, start_ref: VerseRef, end_ref: VerseRef):
        super().__init__(start_ref, end_ref)
        self._text = ""
        self._verses: List[Verse] = []

    def add_verse(self, verse: Verse) -> None:
        self._verses.append(verse)
        self._text += verse.text + " "

    def build(self) -> SegmentedPassage:
        return SegmentedPassage(self.start_ref, self.end_ref, self._text.replace("  ", " ").strip(), self._verses)


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


class WordAlignments:
    def __init__(self, aligned_pairs: Collection[AlignedWordPair]):
        self._aligned_pairs = aligned_pairs
        self._target_tokens_by_source_token: Dict[int, List[int]] = self._create_source_to_target_alignment_lookup(
            aligned_pairs
        )
        self._cached_crossed_alignments: Dict[tuple[int, int], int] = {}

    def _create_source_to_target_alignment_lookup(
        self, aligned_pairs: Collection[AlignedWordPair]
    ) -> Dict[int, List[int]]:
        target_tokens_by_source_token: Dict[int, List[int]] = defaultdict(list)
        for aligned_word_pair in aligned_pairs:
            target_tokens_by_source_token[aligned_word_pair.source_index].append(aligned_word_pair.target_index)
        return target_tokens_by_source_token

    def get_aligned_words(self, source_word_index: int) -> List[int]:
        if source_word_index not in self._target_tokens_by_source_token:
            return []
        return self._target_tokens_by_source_token.get(source_word_index)

    def get_num_crossed_alignments(self, src_word_index: int, trg_word_index: int) -> int:
        if (src_word_index, trg_word_index) in self._cached_crossed_alignments:
            return self._cached_crossed_alignments[(src_word_index, trg_word_index)]
        num_crossings = 0
        for aligned_word_pair in self._aligned_pairs:
            if (
                aligned_word_pair.source_index < src_word_index and aligned_word_pair.target_index > trg_word_index
            ) or (aligned_word_pair.source_index > src_word_index and aligned_word_pair.target_index < trg_word_index):
                num_crossings += 1
        self._cached_crossed_alignments[(src_word_index, trg_word_index)] = num_crossings
        return num_crossings


class VerseSegmenter(ABC):
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

    @abstractmethod
    def predict_target_verse_token_offsets(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ) -> List[int]: ...

    def segment_verses(
        self,
        references: List[VerseRef],
        source_tokens: List[str],
        target_tokens: List[str],
        target_text: str,
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ) -> List[Verse]:
        target_verse_offsets: List[int] = self.predict_target_verse_token_offsets(
            source_tokens, target_tokens, source_verse_token_offsets, word_alignments
        )
        return self._create_target_verses_from_offsets(references, target_tokens, target_verse_offsets, target_text)

    def _create_target_verses_from_offsets(
        self,
        references: List[VerseRef],
        target_tokens: List[str],
        target_verse_offsets: List[int],
        target_text: str,
    ) -> SegmentedPassage:
        target_verses = []

        # Special case where passage is a single verse
        if len(target_verse_offsets) == 0:
            verse_ref = references[0]
            target_verses.append(Verse(verse_ref, target_text))
            return self._adjust_verse_boundaries(target_verses)

        current_verse_starting_char_index = 0
        current_verse_ending_char_index = 0
        current_verse_offset_index = 0
        for target_word_index, target_word in enumerate(target_tokens):
            if target_verse_offsets[current_verse_offset_index] == -1:
                verse_ref = references[current_verse_offset_index]
                target_verses.append(Verse(verse_ref, ""))

                current_verse_starting_char_index = current_verse_ending_char_index
                current_verse_offset_index += 1
                if current_verse_offset_index >= len(target_verse_offsets):
                    break

            elif target_word_index == target_verse_offsets[current_verse_offset_index]:
                verse_ref = references[current_verse_offset_index]
                verse_text = target_text[current_verse_starting_char_index:current_verse_ending_char_index]
                target_verses.append(Verse(verse_ref, verse_text))

                current_verse_starting_char_index = current_verse_ending_char_index
                current_verse_offset_index += 1
                if current_verse_offset_index >= len(target_verse_offsets):
                    break

            current_verse_ending_char_index = target_text.index(target_word, current_verse_ending_char_index) + len(
                target_word
            )

        while current_verse_offset_index < len(references):
            last_verse_ref = references[current_verse_offset_index]
            last_verse_text = target_text[current_verse_starting_char_index:]
            target_verses.append(Verse(last_verse_ref, last_verse_text))

            current_verse_starting_char_index = len(target_text)
            current_verse_offset_index += 1

        return self._adjust_verse_boundaries(target_verses)

    def _adjust_verse_boundaries(self, target_verses: List[Verse]) -> List[Verse]:
        for verse, next_verse in zip(target_verses[:-1], target_verses[1:]):
            while len(next_verse.text) > 0 and next_verse.text[0] in self.PROHIBITED_VERSE_STARTING_CHARACTERS:
                verse.text += next_verse.text[0]
                next_verse.text = next_verse.text[1:]
            while len(verse.text) > 0 and verse.text[-1] in self.PROHIBITED_VERSE_ENDING_CHARACTERS:
                next_verse.text = verse.text[-1] + next_verse.text
                verse.text = verse.text[:-1]
        return target_verses


class FewestCrossedAlignmentsVerseSegmenter(VerseSegmenter):
    def predict_target_verse_token_offsets(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ) -> List[int]:
        last_target_verse_offset = -1
        target_verse_offsets: List[int] = []
        for verse_token_offset in source_verse_token_offsets[:-1]:
            if verse_token_offset == len(source_tokens):
                target_verse_offsets.append(len(target_tokens))
                continue

            fewest_crossed_alignments = 1_000_000
            best_split_indices = []
            for trg_word_index in range(last_target_verse_offset + 1, len(target_tokens)):
                crossed_alignments = word_alignments.get_num_crossed_alignments(verse_token_offset - 1, trg_word_index)

                if trg_word_index < len(target_tokens):
                    crossed_alignments += word_alignments.get_num_crossed_alignments(verse_token_offset, trg_word_index)

                if crossed_alignments == fewest_crossed_alignments:
                    best_split_indices.append(trg_word_index)
                elif crossed_alignments < fewest_crossed_alignments:
                    fewest_crossed_alignments = crossed_alignments
                    best_split_indices = [trg_word_index]

            if len(best_split_indices) == 0:
                target_verse_offsets.append(-1)
            else:
                if verse_token_offset > 0 and not contains_letter(source_tokens[verse_token_offset - 1]):
                    aligned_words = word_alignments.get_aligned_words(verse_token_offset - 1)
                    if (
                        len(aligned_words) == 1
                        and not contains_letter(target_tokens[aligned_words[0]])
                        and aligned_words[0] in best_split_indices
                    ):
                        target_verse_offsets.append(aligned_words[0])
                        last_target_verse_offset = target_verse_offsets[-1]
                        continue
                if not contains_letter(source_tokens[verse_token_offset]):
                    aligned_words = word_alignments.get_aligned_words(verse_token_offset)
                    if (
                        len(aligned_words) == 1
                        and not contains_letter(target_tokens[aligned_words[0]])
                        and aligned_words[0] in best_split_indices
                    ):
                        target_verse_offsets.append(aligned_words[0])
                        last_target_verse_offset = target_verse_offsets[-1]
                        continue
                if len(best_split_indices) > 1:
                    best_split_index = best_split_indices[0]
                    for split_index in best_split_indices:
                        if not contains_letter(target_tokens[split_index-1]):
                            best_split_index = split_index
                            break
                    target_verse_offsets.append(best_split_index)
                else:
                    target_verse_offsets.append(best_split_indices[0])
            last_target_verse_offset = target_verse_offsets[-1]
        return target_verse_offsets


class AdaptedMarkerPlacementVerseSegmenter(VerseSegmenter):
    def predict_target_verse_token_offsets(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ) -> List[int]:
        return [
            self._predict_single_verse_token_offset(
                source_tokens, target_tokens, source_verse_token_offset, word_alignments
            )
            for source_verse_token_offset in source_verse_token_offsets[:-1]
        ]

    def _predict_single_verse_token_offset(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offset: int,
        word_alignments: WordAlignments,
    ) -> int:
        # If the token on either side of a potential target location is punctuation,
        # use it as the basis for deciding the target marker location
        trg_hyp = -1
        punct_hyps = [-1, 0]
        for punct_hyp in punct_hyps:
            src_hyp = source_verse_token_offset + punct_hyp
            if src_hyp < 0 or src_hyp >= len(source_tokens):
                continue
            # Only accept aligned pairs where both the src and trg token are punctuation
            hyp_tok = source_tokens[src_hyp]
            if len(hyp_tok) > 0 and not any(c.isalpha() for c in hyp_tok) and src_hyp < len(source_tokens):
                aligned_trg_toks = word_alignments.get_aligned_words(src_hyp)
                # If aligning to a token that precedes that marker,
                # the trg token predicted to be closest to the marker
                # is the last token aligned to the src rather than the first
                for trg_idx in reversed(aligned_trg_toks) if punct_hyp < 0 else aligned_trg_toks:
                    trg_tok = target_tokens[trg_idx]
                    if len(trg_tok) > 0 and not any(c.isalpha() for c in trg_tok):
                        trg_hyp = trg_idx
                        break
            if trg_hyp != -1:
                # Since the marker location is represented by the token after the marker,
                # adjust the index when aligning to punctuation that precedes the token
                return trg_hyp + (1 if punct_hyp == -1 else 0)

        hyps = [0, 1, 2]
        best_hyp = -1
        best_num_crossings = 200**2  # mostly meaningless, a big number
        checked = set()
        for hyp in hyps:
            src_hyp = source_verse_token_offset + hyp
            if src_hyp in checked:
                continue
            trg_hyp = -1
            while trg_hyp == -1 and src_hyp >= 0 and src_hyp < len(source_tokens):
                checked.add(src_hyp)
                aligned_trg_toks = word_alignments.get_aligned_words(src_hyp)
                if len(aligned_trg_toks) > 0:
                    # If aligning with a source token that precedes the marker,
                    # the target token predicted to be closest to the marker is the last aligned token rather than the first
                    trg_hyp = aligned_trg_toks[-1 if hyp < 0 else 0]
                else:  # continue the search outwards
                    src_hyp += -1 if hyp < 0 else 1
            if trg_hyp != -1:
                num_crossings = word_alignments.get_num_crossed_alignments(source_verse_token_offset, trg_hyp)
                if num_crossings < best_num_crossings:
                    best_hyp = trg_hyp
                    best_num_crossings = num_crossings
                if num_crossings == 0:
                    break

        # If no alignments found, insert at the end of the sentence
        return best_hyp if best_hyp != -1 else len(target_tokens)


class ParallelPassage(VerseRange):

    def __init__(
        self,
        start_ref: VerseRef,
        end_ref: VerseRef,
        source_verses: List[Verse],
        target_text: str,
        word_alignments: WordAlignments,
        tokenized_source_verses: Optional[List[List[str]]] = None,
        tokenized_source_text: Optional[List[str]] = None,
        tokenized_target_text: Optional[List[str]] = None,
        verse_token_offsets: Optional[List[int]] = None,
    ):
        super().__init__(start_ref, end_ref)
        self._start_ref = start_ref
        self._end_ref = end_ref
        self._source_verses = source_verses
        self._target_text = target_text
        self._word_alignments = word_alignments

        if tokenized_source_verses is not None:
            self._tokenized_source_verses = tokenized_source_verses
        else:
            self._tokenized_source_verses: List[List[str]] = []

        if tokenized_source_text is not None:
            self._tokenized_source_text = tokenized_source_text
        else:
            self._tokenized_source_text = []

        if tokenized_target_text is not None:
            self._tokenized_target_text = tokenized_target_text
        else:
            self._tokenized_target_text = []

        if verse_token_offsets is not None:
            self._source_verse_token_offsets = verse_token_offsets
        else:
            self._source_verse_token_offsets = []

    def segment_target_passage(self, verse_segmenter: VerseSegmenter) -> SegmentedPassage:
        target_verses: List[Verse] = verse_segmenter.segment_verses(
            [verse.reference for verse in self._source_verses],
            self._tokenized_source_text,
            self._tokenized_target_text,
            self._target_text,
            self._source_verse_token_offsets,
            self._word_alignments,
        )
        return SegmentedPassage(self._start_ref, self._end_ref, self._target_text, target_verses)

    def get_source_passage(self) -> SegmentedPassage:
        return SegmentedPassage(
            self._start_ref, self._end_ref, " ".join([v.text for v in self._source_verses]), self._source_verses
        )


class ParallelPassageBuilder(VerseCollector):
    tokenizer = LatinWordTokenizer()

    def __init__(self, start_ref: VerseRef, end_ref: VerseRef, target_text: str):
        super().__init__(start_ref, end_ref)
        self._target_text = target_text
        self._tokenized_target_text = list(self.tokenizer.tokenize(self._target_text))
        self._source_verses: List[Verse] = []
        self._word_alignments: WordAlignments
        self._tokenized_source_verses: List[List[str]] = []
        self._tokenized_source_text: List[str] = []
        self._verse_token_offsets: List[int] = []

    def add_verse(self, verse: Verse) -> None:
        self._source_verses.append(verse)
        tokens = list(self.tokenizer.tokenize(verse.text))
        self._tokenized_source_verses.append(tokens)
        self._tokenized_source_text.extend(tokens)
        self._verse_token_offsets.append(len(self._tokenized_source_text))

    def get_token_separated_source_text_for_alignment(self) -> str:
        return " ".join(self._tokenized_source_text)

    def get_token_separated_target_text_for_alignment(self) -> str:
        return " ".join(self._tokenized_target_text)

    def set_word_alignments(self, word_alignments: WordAlignments) -> None:
        self._word_alignments = word_alignments

    def build(self) -> ParallelPassage:
        if self._word_alignments is None:
            raise ValueError("Word alignments not loaded")
        return ParallelPassage(
            self.start_ref,
            self.end_ref,
            self._source_verses,
            self._target_text,
            self._word_alignments,
            self._tokenized_source_verses,
            self._tokenized_source_text,
            self._tokenized_target_text,
            self._verse_token_offsets,
        )


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
    def generate(self, source_passages: List[str], target_passages: List[str]) -> Generator[WordAlignments, None, None]:
        pass


class EflomalAlignmentGenerator(AlignmentGenerator):
    def __init__(self, save_alignments: bool = False, target_passage_file: Optional[Path] = None):
        self._save_alignments = save_alignments
        self._target_passage_file = target_passage_file

    def generate(self, source_passages: List[str], target_passages: List[str]) -> Generator[WordAlignments, None, None]:
        with TemporaryDirectory() as td:
            align_path = Path(td, "sym-align.txt")
            src_path = Path(td, "src_align.txt")
            trg_path = Path(td, "trg_align.txt")
            write_corpus(src_path, source_passages)
            write_corpus(trg_path, target_passages)
            compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), "eflomal", align_path)

            for line in load_corpus(align_path):
                yield WordAlignments(AlignedWordPair.from_string(line))

            if self._save_alignments:
                assert self._target_passage_file is not None
                saved_alignments_file = self._target_passage_file.with_suffix(".alignments.txt")
                shutil.copy(align_path, saved_alignments_file)


class SavedAlignmentGenerator(AlignmentGenerator):
    def __init__(self, alignment_file: Path):
        self._alignment_file = alignment_file

    def generate(self, source_passages: List[str], target_passages: List[str]) -> Generator[WordAlignments, None, None]:
        for alignment_line in load_corpus(self._alignment_file):
            yield WordAlignments(AlignedWordPair.from_string(alignment_line))


class ParatextProjectReader:
    def __init__(self, project_name: str):
        self._project_name = project_name

    def collect_verses(self, verse_collectors: List[VerseCollector]) -> List[VerseCollector]:
        settings = FileParatextProjectSettingsParser(SIL_NLP_ENV.pt_projects_dir / self._project_name).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        for book in self._get_all_required_books(verse_collectors):
            usfm_text = UsfmFileText(
                stylesheet,
                encoding,
                book,
                SIL_NLP_ENV.pt_projects_dir / self._project_name / settings.get_book_file_name(book),
            )
            for row in usfm_text:
                for verse_collector in verse_collectors:
                    if isinstance(row.ref, ScriptureRef) and verse_collector.is_ref_in_range(row.ref.verse_ref):
                        verse_collector.add_verse(Verse(row.ref.verse_ref, row.text))
        return verse_collectors

    def _get_all_required_books(self, verse_collectors: List[VerseCollector]) -> Set[str]:
        all_books: Set[str] = set()
        for verse_collector in verse_collectors:
            all_books.add(verse_collector.start_ref.book)
            all_books.add(verse_collector.end_ref.book)
        return all_books


class ParallelPassageCollection:
    def __init__(self, source_project_name: str, target_passage_file: Path, alignment_generator: AlignmentGenerator):
        target_passages = PassageReader(target_passage_file).get_passages()
        parallel_passage_builders = self._collect_parallel_passages(source_project_name, target_passages)
        self._create_word_alignment_matrix(parallel_passage_builders, alignment_generator)
        self._parallel_passages = [
            parallel_passage_builder.build() for parallel_passage_builder in parallel_passage_builders
        ]

    def _collect_parallel_passages(
        self, source_project_name: str, target_passages: List[Passage]
    ) -> List[ParallelPassageBuilder]:
        paratext_project_reader = ParatextProjectReader(source_project_name)
        parallel_passage_builders = [ParallelPassageBuilder(t.start_ref, t.end_ref, t.text) for t in target_passages]
        return paratext_project_reader.collect_verses(parallel_passage_builders)

    def _create_word_alignment_matrix(
        self, parallel_passage_builders: List[ParallelPassageBuilder], alignment_generator: AlignmentGenerator
    ) -> None:
        for index, word_alignments in enumerate(
            alignment_generator.generate(
                [
                    passage_builder.get_token_separated_source_text_for_alignment()
                    for passage_builder in parallel_passage_builders
                ],
                [
                    passage_builder.get_token_separated_target_text_for_alignment()
                    for passage_builder in parallel_passage_builders
                ],
            )
        ):
            parallel_passage_builders[index].set_word_alignments(word_alignments)

    def segment_target_passages(self, verse_segmenter: VerseSegmenter) -> Iterable[SegmentedPassage]:
        for passage in self._parallel_passages:
            yield passage.segment_target_passage(verse_segmenter)

    def get_source_segmented_passages(self) -> List[SegmentedPassage]:
        return [parallel_passage.get_source_passage() for parallel_passage in self._parallel_passages]


class ReferenceVerseSegmentationReader:
    def read_passages(self, target_project_name: str, target_passage_file: Path):
        passages = PassageReader(target_passage_file).get_passages()
        return self._read_segmented_passages(passages, target_project_name)

    def _read_segmented_passages(self, passages: List[Passage], target_project_name: str) -> List[SegmentedPassage]:
        paratext_project_reader = ParatextProjectReader(target_project_name)
        segmented_passage_builders: List[SegmentedPassageBuilder] = [
            SegmentedPassageBuilder(p.start_ref, p.end_ref) for p in passages
        ]
        paratext_project_reader.collect_verses(segmented_passage_builders)
        return [segmented_passage_builder.build() for segmented_passage_builder in segmented_passage_builders]


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
        "--use-saved-alignments", help="Use pre-computed alignments from a previous run", default=None, action="store_true"
    )
    args = parser.parse_args()

    parallel_passages = ParallelPassageCollectionFactory(args.save_alignments, args.use_saved_alignments).create(
        args.source_project, Path(args.target_passages)
    )
    src_segmented_passages = parallel_passages.get_source_segmented_passages()

    verse_segmenter = FewestCrossedAlignmentsVerseSegmenter()
    trg_segmented_passages = list(parallel_passages.segment_target_passages(verse_segmenter))

    src_path = Path(args.target_passages).with_suffix(".src.txt")
    trg_path = Path(args.target_passages).with_suffix(".trg.txt")
    with open(src_path, "w", encoding="utf-8") as src_output, open(trg_path, "w", encoding="utf-8") as trg_output:
        for src_passage, trg_passage in zip(src_segmented_passages, trg_segmented_passages):
            src_passage.write_to_file(src_output)
            trg_passage.write_to_file(trg_output)

    if args.compare_against is not None:
        reference_segmentations = ReferenceVerseSegmentationReader().read_passages(
            args.compare_against, Path(args.target_passages)
        )
        segmentation_evaluator = SegmentationEvaluator(reference_segmentations)
        segmentation_evaluation = segmentation_evaluator.evaluate_segmentation(trg_segmented_passages)
        print(
            f"Segmentation accuracy = {segmentation_evaluation.get_accuracy() * 100}%, average distance = {segmentation_evaluation.get_average_distance()} characters"
        )


if __name__ == "__main__":
    main()
