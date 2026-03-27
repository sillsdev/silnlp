import argparse
import json
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Collection, Dict, Generator, List, MutableSet, Optional, TextIO, Tuple, TypeVar

import regex
from machine.corpora import AlignedWordPair, FileParatextProjectSettingsParser, ScriptureRef, UsfmFileText
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef
from machine.tokenization import LatinWordTokenizer
from numpy import exp, sqrt
from pyparsing import Iterable

from silnlp.common.environment import SIL_NLP_ENV

from ..common.corpus import load_corpus, write_corpus
from .utils import compute_alignment_scores

unicode_letter_regex = regex.compile("\\p{L}")


def contains_letter(token: str) -> bool:
    return unicode_letter_regex.search(token) is not None


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
    def __init__(self, src_length: int, trg_length: int, aligned_pairs: Collection[AlignedWordPair]):
        self._src_length = src_length
        self._trg_length = trg_length
        self._aligned_pairs = aligned_pairs
        self._target_tokens_by_source_token: Dict[int, List[int]] = self._create_source_to_target_alignment_lookup(
            aligned_pairs
        )
        self._source_tokens_by_target_token: Dict[int, List[int]] = self._create_target_to_source_alignment_lookup(
            aligned_pairs
        )
        self._cached_crossed_alignments: Dict[tuple[int, int, float], float] = {}

    def _create_source_to_target_alignment_lookup(
        self, aligned_pairs: Collection[AlignedWordPair]
    ) -> Dict[int, List[int]]:
        target_tokens_by_source_token: Dict[int, List[int]] = defaultdict(list)
        for aligned_word_pair in aligned_pairs:
            target_tokens_by_source_token[aligned_word_pair.source_index].append(aligned_word_pair.target_index)
        return target_tokens_by_source_token

    def _create_target_to_source_alignment_lookup(
        self, aligned_pairs: Collection[AlignedWordPair]
    ) -> Dict[int, List[int]]:
        source_tokens_by_target_token: Dict[int, List[int]] = defaultdict(list)
        for aligned_word_pair in aligned_pairs:
            source_tokens_by_target_token[aligned_word_pair.target_index].append(aligned_word_pair.source_index)
        return source_tokens_by_target_token

    def get_target_aligned_words(self, source_word_index: int) -> List[int]:
        return self._target_tokens_by_source_token.get(source_word_index) or []

    def get_source_aligned_words(self, target_word_index: int) -> List[int]:
        return self._source_tokens_by_target_token.get(target_word_index) or []

    def get_num_crossed_alignments(
        self, src_word_index: int, trg_word_index: int, off_diagonal_penalty: float = 0
    ) -> float:
        if (src_word_index, trg_word_index, off_diagonal_penalty) in self._cached_crossed_alignments:
            return self._cached_crossed_alignments[(src_word_index, trg_word_index, off_diagonal_penalty)]
        num_crossings = 0
        for aligned_word_pair in self._aligned_pairs:
            # By convention, a break at "word_index" is placed immediately before
            # the word at words[word_index]
            if (
                aligned_word_pair.source_index < src_word_index and aligned_word_pair.target_index >= trg_word_index
            ) or (aligned_word_pair.source_index >= src_word_index and aligned_word_pair.target_index < trg_word_index):
                num_crossings += 1 * (1 - off_diagonal_penalty) + off_diagonal_penalty * (
                    1 - self._get_off_diagonal_probability(src_word_index, trg_word_index)
                )

        self._cached_crossed_alignments[(src_word_index, trg_word_index, off_diagonal_penalty)] = num_crossings
        return num_crossings

    def _get_off_diagonal_probability(self, src_word_index: int, trg_word_index: int) -> float:
        expected_trg_index = int(src_word_index * self._trg_length / self._src_length)
        distance_from_diagonal = abs(trg_word_index - expected_trg_index)
        return 15.17 / sqrt(2 * 3.14 * 35) * exp(-0.5 * (distance_from_diagonal) ** 2 / 35)

    def remove_links_crossing_n(self, n: int, other_alignment: "WordAlignments") -> "WordAlignments":
        pairs_to_retain: List[AlignedWordPair] = []
        for aligned_pair in self._aligned_pairs:
            if other_alignment.get_num_crossed_alignments(aligned_pair.source_index, aligned_pair.target_index) < n:
                pairs_to_retain.append(aligned_pair)
        return WordAlignments(self._src_length, self._trg_length, pairs_to_retain)

    def append_to_file(self, output_file: Path) -> None:
        with output_file.open("w") as f:
            for aligned_pair in self._aligned_pairs:
                f.write(f"{aligned_pair.source_index}-{aligned_pair.target_index} ")
            f.write("\n")

    def to_json(self) -> Dict[str, Any]:
        return {
            "src_length": self._src_length,
            "trg_length": self._trg_length,
            "aligned_pairs": " ".join(
                [f"{aligned_pair.source_index}-{aligned_pair.target_index}" for aligned_pair in self._aligned_pairs]
            ),
        }

    @classmethod
    def from_json(cls, word_alignment_json: Dict[str, Any]) -> "WordAlignments":
        src_length = word_alignment_json["src_length"]
        trg_length = word_alignment_json["trg_length"]
        aligned_pairs = []
        for pair_str in word_alignment_json["aligned_pairs"].split():
            source_index, target_index = map(int, pair_str.split("-"))
            aligned_pairs.append(AlignedWordPair(source_index, target_index))
        return cls(src_length=src_length, trg_length=trg_length, aligned_pairs=aligned_pairs)


class WordAlignmentsBuilder:
    def __init__(self, src_length: int, trg_length: int):
        self._src_length = src_length
        self._trg_length = trg_length
        self._aligned_pairs: List[AlignedWordPair] = []

    def add_aligments(self, aligned_pairs: Collection[AlignedWordPair]) -> None:
        self._aligned_pairs.extend(aligned_pairs)

    def build(self) -> WordAlignments:
        return WordAlignments(self._src_length, self._trg_length, self._aligned_pairs)


class VerseOffsetPredictor(ABC):
    def __init__(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ):
        self._source_tokens = source_tokens
        self._target_tokens = target_tokens
        self._source_verse_token_offsets = source_verse_token_offsets
        self._word_alignments = word_alignments

    @abstractmethod
    def predict_target_verse_token_offsets(self) -> List[int]: ...


class AbstractVerseOffsetPredictorFactory:
    @abstractmethod
    def create(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
        sub_passages: Optional[List["SubPassage"]] = None,
    ) -> VerseOffsetPredictor: ...


class VerseSegmenter(ABC):
    # The characters » and › are not in this list since they often
    # start verses as quote continuers in Spanish
    _PROHIBITED_VERSE_STARTING_CHARACTERS: Set[str] = {
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
        "”",
        "’",
    }
    _PROHIBITED_VERSE_ENDING_CHARACTERS: Set[str] = {"(", "[", "{", "«", "‹", "“", "‘"}
    _PUNCTUATION_AND_SENTENCE_STARTING_PATTERN: regex.Pattern = regex.compile(
        r".*([^\w\s])\s*(\p{Lu}\w+(\s+\w+)?(\s+\w+)?)\s*$"
    )
    _WORDS_AND_SENTENCE_ENDING_PATTERN: regex.Pattern = regex.compile(
        r"^(\p{Ll}\w+(\s+\w+)?(\s+\w+)?)([\.,;:!\?\)\]”’][”’]*)\s*"
    )

    def __init__(
        self,
        verse_offset_predictor_factory: AbstractVerseOffsetPredictorFactory,
        source_tokens: List[str],
        target_tokens: List[str],
        target_text: str,
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
        sub_passages: Optional[List["SubPassage"]] = None,
    ):
        self._verse_offset_predictor_factory = verse_offset_predictor_factory
        self._source_tokens = source_tokens
        self._target_tokens = target_tokens
        self._target_text = target_text
        self._source_verse_token_offsets = source_verse_token_offsets
        self._word_alignments = word_alignments
        self._sub_passages = sub_passages if sub_passages is not None else []

    def segment_verses(self, references: List[VerseRef]) -> List[Verse]:
        if self._sub_passages is not None and len(self._sub_passages) > 0:
            target_verse_offsets = self._get_target_verse_offsets_from_sub_passages()
        else:
            target_verse_offsets: List[int] = self._verse_offset_predictor_factory.create(
                self._source_tokens, self._target_tokens, self._source_verse_token_offsets, self._word_alignments
            ).predict_target_verse_token_offsets()
        return self._create_target_verses_from_offsets(references, target_verse_offsets)

    def _get_target_verse_offsets_from_sub_passages(self) -> List[int]:
        target_verse_offsets = []
        cumulative_token_offset = 0
        for sub_passage in self._sub_passages:
            if sub_passage.source_verse_token_offsets is not None and len(sub_passage.source_verse_token_offsets) > 0:
                for sub_passage_predicted_offset in self._verse_offset_predictor_factory.create(
                    sub_passage.source_tokens,
                    sub_passage.target_tokens,
                    sub_passage.source_verse_token_offsets,
                    sub_passage.word_alignments,
                ).predict_target_verse_token_offsets():
                    target_verse_offsets.append(sub_passage_predicted_offset + cumulative_token_offset)
            cumulative_token_offset += len(sub_passage.target_tokens)
        return target_verse_offsets

    def _create_target_verses_from_offsets(
        self,
        references: List[VerseRef],
        target_verse_offsets: List[int],
    ) -> List[Verse]:
        target_verses = []

        # Special case where passage is a single verse
        if len(target_verse_offsets) == 0:
            verse_ref = references[0]
            target_verses.append(Verse(verse_ref, self._target_text))
            return self._adjust_verse_boundaries(target_verses)

        current_verse_starting_char_index = 0
        current_verse_ending_char_index = 0
        current_verse_offset_index = 0
        for target_word_index, target_word in enumerate(self._target_tokens):
            if (
                target_verse_offsets[current_verse_offset_index] == -1
                or target_word_index >= target_verse_offsets[current_verse_offset_index]
            ):
                verse_ref = references[current_verse_offset_index]
                verse_text = self._target_text[current_verse_starting_char_index:current_verse_ending_char_index]
                target_verses.append(Verse(verse_ref, verse_text))

                current_verse_starting_char_index = current_verse_ending_char_index
                current_verse_offset_index += 1
                if current_verse_offset_index >= len(target_verse_offsets):
                    break

            current_verse_ending_char_index = self._target_text.index(
                target_word, current_verse_ending_char_index
            ) + len(target_word)

        while current_verse_offset_index < len(references):
            last_verse_ref = references[current_verse_offset_index]
            last_verse_text = self._target_text[current_verse_starting_char_index:]
            target_verses.append(Verse(last_verse_ref, last_verse_text))

            current_verse_starting_char_index = len(self._target_text)
            current_verse_offset_index += 1

        return self._adjust_verse_boundaries(target_verses)

    def _adjust_verse_boundaries(self, target_verses: List[Verse]) -> List[Verse]:
        for verse, next_verse in zip(target_verses[:-1], target_verses[1:]):
            while len(next_verse.text) > 0 and next_verse.text[0] in self._PROHIBITED_VERSE_STARTING_CHARACTERS:
                verse.text += next_verse.text[0]
                next_verse.text = next_verse.text[1:]
            while len(verse.text) > 0 and verse.text[-1] in self._PROHIBITED_VERSE_ENDING_CHARACTERS:
                next_verse.text = verse.text[-1] + next_verse.text
                verse.text = verse.text[:-1]
            if self._verse_ends_with_start_of_sentence(verse):
                verse, next_verse = self._adjust_for_missed_sentence_start(verse, next_verse)
            if self._verse_starts_with_end_of_sentence(next_verse):
                verse, next_verse = self._adjust_for_late_sentence_end(verse, next_verse)
        return target_verses

    def _verse_ends_with_start_of_sentence(self, verse: Verse) -> bool:
        return self._PUNCTUATION_AND_SENTENCE_STARTING_PATTERN.match(verse.text) is not None

    def _adjust_for_missed_sentence_start(self, verse: Verse, next_verse: Verse) -> Tuple[Verse, Verse]:
        match = self._PUNCTUATION_AND_SENTENCE_STARTING_PATTERN.match(verse.text)
        if match is not None:
            capitalized_word = match.group(2)
            verse.text = verse.text[: match.end(1)]
            next_verse.text = capitalized_word + " " + next_verse.text
        return verse, next_verse

    def _verse_starts_with_end_of_sentence(self, verse: Verse) -> bool:
        return self._WORDS_AND_SENTENCE_ENDING_PATTERN.match(verse.text) is not None

    def _adjust_for_late_sentence_end(self, verse: Verse, next_verse: Verse) -> Tuple[Verse, Verse]:
        match = self._WORDS_AND_SENTENCE_ENDING_PATTERN.match(next_verse.text)
        if match is not None:
            words = match.group(1)
            punctuation = match.group(4)
            verse.text = verse.text + words + punctuation
            next_verse.text = next_verse.text[match.end(0) :]
        return verse, next_verse


class FewestCrossedAlignmentsVerseOffsetPredictor(VerseOffsetPredictor):
    def __init__(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
        psuedoalignment_weight: float,
        psuedoalignment_exponent: float,
    ):
        super().__init__(source_tokens, target_tokens, source_verse_token_offsets, word_alignments)
        num_src_tokens = len(self._source_tokens)
        num_trg_tokens = len(self._target_tokens)

        # segmentation breaks are selected greedily
        # these dicts keep track of the nearest segment break for each position
        self._nearest_preceding_segment_break = [(0, 0) for _ in range(num_src_tokens + 1)]
        self._nearest_trailing_segment_break = [(num_src_tokens, num_trg_tokens) for _ in range(num_src_tokens + 1)]

        self._remaining_breaks_to_map: MutableSet[int] = set(
            [i for i in self._source_verse_token_offsets if i != num_src_tokens]
        )
        self._target_verse_token_offsets: List[int] = [-1 for i in self._source_verse_token_offsets]

        if self._source_verse_token_offsets[-1] == num_src_tokens:
            self._target_verse_token_offsets[-1] = num_trg_tokens

        self._pseudoalignment_weight = psuedoalignment_weight
        self._pseudoalignment_exponent = psuedoalignment_exponent

    def _add_trg_verse_break(self, src_offset: int, trg_offset: int) -> None:
        self._remaining_breaks_to_map.remove(src_offset)
        for index in range(len(self._source_verse_token_offsets)):
            if self._source_verse_token_offsets[index] == src_offset:
                self._target_verse_token_offsets[index] = trg_offset

        for src_index in reversed(range(0, src_offset)):
            if self._nearest_trailing_segment_break[src_index][0] < src_offset:
                break
            self._nearest_trailing_segment_break[src_index] = (src_offset, trg_offset)

        for src_index in range(src_offset, len(self._nearest_preceding_segment_break)):
            if self._nearest_preceding_segment_break[src_index][0] > src_offset:
                break
            self._nearest_preceding_segment_break[src_index] = (src_offset, trg_offset)

    def predict_target_verse_token_offsets(self) -> List[int]:

        # greedily select the best segment break
        while len(self._remaining_breaks_to_map) > 0:
            lowest_crossed_alignment_weight = 1_000_000
            best_split_indices: List[Tuple[int, int]] = []
            for src_break_offset in self._source_verse_token_offsets:
                if (
                    src_break_offset == len(self._source_tokens) + 1
                    or src_break_offset not in self._remaining_breaks_to_map
                ):
                    continue
                for trg_break_offset in range(
                    self._nearest_preceding_segment_break[src_break_offset][1] + 1,
                    self._nearest_trailing_segment_break[src_break_offset][1] - 1,
                ):
                    crossed_alignments = self._calculate_crossed_alignment_weight(src_break_offset, trg_break_offset)

                    if crossed_alignments == lowest_crossed_alignment_weight:
                        best_split_indices.append((src_break_offset, trg_break_offset))
                    elif crossed_alignments < lowest_crossed_alignment_weight:
                        lowest_crossed_alignment_weight = crossed_alignments
                        best_split_indices = [(src_break_offset, trg_break_offset)]

            if len(best_split_indices) == 0:
                break
            else:
                src_break_offset, trg_break_offset = best_split_indices[0]
                if src_break_offset > 0 and not contains_letter(self._source_tokens[src_break_offset - 1]):
                    aligned_words = self._word_alignments.get_target_aligned_words(src_break_offset - 1)
                    if (
                        len(aligned_words) == 1
                        and not contains_letter(self._target_tokens[aligned_words[0]])
                        and (src_break_offset, aligned_words[0]) in best_split_indices
                    ):
                        adjusted_trg_break_offset = aligned_words[0]
                        source_aligned_words = self._word_alignments.get_source_aligned_words(
                            adjusted_trg_break_offset + 1
                        )
                        while (
                            len(source_aligned_words) == 0
                            and adjusted_trg_break_offset + 1 < len(self._target_tokens)
                            and self._target_tokens[adjusted_trg_break_offset + 1] == '"'
                        ):
                            adjusted_trg_break_offset += 1
                            source_aligned_words = self._word_alignments.get_source_aligned_words(
                                adjusted_trg_break_offset
                            )
                        self._add_trg_verse_break(src_break_offset, adjusted_trg_break_offset + 1)
                        continue
                if src_break_offset < len(self._source_tokens) and not contains_letter(
                    self._source_tokens[src_break_offset]
                ):
                    aligned_words = self._word_alignments.get_target_aligned_words(src_break_offset)
                    if (
                        len(aligned_words) == 1
                        and not contains_letter(self._target_tokens[aligned_words[0]])
                        and (src_break_offset, aligned_words[0] + 1) in best_split_indices
                    ):
                        self._add_trg_verse_break(src_break_offset, aligned_words[0] + 1)
                        continue

            if len(best_split_indices) > 1:
                best_src_break_offset, best_trg_break_offset = best_split_indices[0]
                for src_break_offset, trg_break_offset in best_split_indices:
                    if not contains_letter(self._target_tokens[trg_break_offset - 1]):
                        best_src_break_offset = src_break_offset
                        best_trg_break_offset = trg_break_offset
                        break
                self._add_trg_verse_break(best_src_break_offset, best_trg_break_offset)
            else:
                self._add_trg_verse_break(best_split_indices[0][0], best_split_indices[0][1])
        return self._target_verse_token_offsets

    def _calculate_crossed_alignment_weight(self, src_word_index: int, trg_word_index: int) -> float:
        actual_crossed_alignments = float(
            self._word_alignments.get_num_crossed_alignments(src_word_index, trg_word_index)
        )
        crossed_pseudoalignment_weight = self._calculate_crossed_pseudoalignment_weight(src_word_index, trg_word_index)
        return self._combine_alignment_scores(actual_crossed_alignments, crossed_pseudoalignment_weight)

    def _calculate_crossed_pseudoalignment_weight(self, src_word_index: int, trg_word_index: int) -> float:
        matrix_src_width = (
            self._nearest_trailing_segment_break[src_word_index][0]
            - self._nearest_preceding_segment_break[src_word_index][0]
        )
        matrix_trg_width = (
            self._nearest_trailing_segment_break[src_word_index][1]
            - self._nearest_preceding_segment_break[src_word_index][1]
        )

        diagonal_src_index_projection = (
            src_word_index - self._nearest_preceding_segment_break[src_word_index][0]
        ) * matrix_trg_width / matrix_src_width + self._nearest_preceding_segment_break[src_word_index][0]
        crossed_alignment_weight = abs(trg_word_index - diagonal_src_index_projection)
        return crossed_alignment_weight**self._pseudoalignment_exponent

    def _combine_alignment_scores(
        self, actual_crossed_alignments: float, crossed_pseudoalignment_weight: float
    ) -> float:
        return (actual_crossed_alignments + self._pseudoalignment_weight * crossed_pseudoalignment_weight) / (
            1 + self._pseudoalignment_weight
        )


class FewestCrossedAlignmentsVerseOffsetPredictorFactory(AbstractVerseOffsetPredictorFactory):
    def __init__(self, pseudoalignment_weight: float, pseudoalignment_exponent: float):
        self._pseudoalignment_weight = pseudoalignment_weight
        self._pseudoalignment_exponent = pseudoalignment_exponent

    def create(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ) -> FewestCrossedAlignmentsVerseOffsetPredictor:
        return FewestCrossedAlignmentsVerseOffsetPredictor(
            source_tokens,
            target_tokens,
            source_verse_token_offsets,
            word_alignments,
            self._pseudoalignment_weight,
            self._pseudoalignment_exponent,
        )


class ScoringFunctionVerseOffsetPredictor(VerseOffsetPredictor):
    def __init__(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
        break_scorer: "BreakScorer",
    ):
        super().__init__(source_tokens, target_tokens, source_verse_token_offsets, word_alignments)
        num_src_tokens = len(self._source_tokens)
        num_trg_tokens = len(self._target_tokens)

        # segmentation breaks are selected greedily
        # these dicts keep track of the nearest segment break for each position
        self._nearest_preceding_segment_break = [(0, 0) for _ in range(num_src_tokens + 1)]
        self._nearest_trailing_segment_break = [(num_src_tokens, num_trg_tokens) for _ in range(num_src_tokens + 1)]

        self._remaining_breaks_to_map: MutableSet[int] = set(
            [i for i in self._source_verse_token_offsets if i != num_src_tokens]
        )
        self._target_verse_token_offsets: List[int] = [-1 for i in self._source_verse_token_offsets]

        if self._source_verse_token_offsets[-1] == num_src_tokens:
            self._target_verse_token_offsets[-1] = num_trg_tokens

        self._break_scorer = break_scorer

    def _add_trg_verse_break(self, src_offset: int, trg_offset: int) -> None:
        self._remaining_breaks_to_map.remove(src_offset)
        for index in range(len(self._source_verse_token_offsets)):
            if self._source_verse_token_offsets[index] == src_offset:
                self._target_verse_token_offsets[index] = trg_offset

        for src_index in reversed(range(0, src_offset)):
            if self._nearest_trailing_segment_break[src_index][0] < src_offset:
                break
            self._nearest_trailing_segment_break[src_index] = (src_offset, trg_offset)

        for src_index in range(src_offset, len(self._nearest_preceding_segment_break)):
            if self._nearest_preceding_segment_break[src_index][0] > src_offset:
                break
            self._nearest_preceding_segment_break[src_index] = (src_offset, trg_offset)

    def predict_target_verse_token_offsets(self) -> List[int]:

        # greedily select the best segment break
        while len(self._remaining_breaks_to_map) > 0:
            best_score = -1_000_000
            best_split_indices: List[Tuple[int, int]] = []
            for src_break_offset in self._source_verse_token_offsets:
                if (
                    src_break_offset == len(self._source_tokens) + 1
                    or src_break_offset not in self._remaining_breaks_to_map
                ):
                    continue
                for trg_break_offset in range(
                    self._nearest_preceding_segment_break[src_break_offset][1] + 1,
                    self._nearest_trailing_segment_break[src_break_offset][1] - 1,
                ):
                    score = self._break_scorer.score_potential_break(src_break_offset, trg_break_offset)
                    if score == best_score:
                        best_split_indices.append((src_break_offset, trg_break_offset))
                    elif score > best_score:
                        best_score = score
                        best_split_indices = [(src_break_offset, trg_break_offset)]

            if len(best_split_indices) == 0:
                break
            else:
                src_break_offset, trg_break_offset = best_split_indices[0]
                if src_break_offset > 0 and not contains_letter(self._source_tokens[src_break_offset - 1]):
                    aligned_words = self._word_alignments.get_target_aligned_words(src_break_offset - 1)
                    if (
                        len(aligned_words) == 1
                        and not contains_letter(self._target_tokens[aligned_words[0]])
                        and (src_break_offset, aligned_words[0]) in best_split_indices
                    ):
                        adjusted_trg_break_offset = aligned_words[0]
                        source_aligned_words = self._word_alignments.get_source_aligned_words(
                            adjusted_trg_break_offset + 1
                        )
                        while (
                            len(source_aligned_words) == 0
                            and adjusted_trg_break_offset + 1 < len(self._target_tokens)
                            and self._target_tokens[adjusted_trg_break_offset + 1] == '"'
                        ):
                            adjusted_trg_break_offset += 1
                            source_aligned_words = self._word_alignments.get_source_aligned_words(
                                adjusted_trg_break_offset
                            )
                        self._add_trg_verse_break(src_break_offset, adjusted_trg_break_offset + 1)
                        continue
                if src_break_offset < len(self._source_tokens) and not contains_letter(
                    self._source_tokens[src_break_offset]
                ):
                    aligned_words = self._word_alignments.get_target_aligned_words(src_break_offset)
                    if (
                        len(aligned_words) == 1
                        and not contains_letter(self._target_tokens[aligned_words[0]])
                        and (src_break_offset, aligned_words[0] + 1) in best_split_indices
                    ):
                        self._add_trg_verse_break(src_break_offset, aligned_words[0] + 1)
                        continue

            if len(best_split_indices) > 1:
                best_src_break_offset, best_trg_break_offset = best_split_indices[0]
                for src_break_offset, trg_break_offset in best_split_indices:
                    if not contains_letter(self._target_tokens[trg_break_offset - 1]):
                        best_src_break_offset = src_break_offset
                        best_trg_break_offset = trg_break_offset
                        break
                self._add_trg_verse_break(best_src_break_offset, best_trg_break_offset)
            else:
                self._add_trg_verse_break(best_split_indices[0][0], best_split_indices[0][1])
        return self._target_verse_token_offsets


class ScoringFunctionVerseOffsetPredictorFactory(AbstractVerseOffsetPredictorFactory):
    def __init__(self, break_scorer_factory: "AbstractBreakScorerFactory"):
        self._break_scorer_factory = break_scorer_factory

    def create(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ) -> ScoringFunctionVerseOffsetPredictor:
        return ScoringFunctionVerseOffsetPredictor(
            source_tokens,
            target_tokens,
            source_verse_token_offsets,
            word_alignments,
            self._break_scorer_factory.create(source_tokens, target_tokens, word_alignments),
        )


class AdaptedMarkerPlacementVerseOffsetPredictor(VerseOffsetPredictor):
    def __init__(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ):
        super().__init__(source_tokens, target_tokens, source_verse_token_offsets, word_alignments)

    def predict_target_verse_token_offsets(self) -> List[int]:
        return [
            self._predict_single_verse_token_offset(
                self._source_tokens, self._target_tokens, source_verse_token_offset, self._word_alignments
            )
            for source_verse_token_offset in self._source_verse_token_offsets[:-1]
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
                aligned_trg_toks = word_alignments.get_target_aligned_words(src_hyp)
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
                aligned_trg_toks = word_alignments.get_target_aligned_words(src_hyp)
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


class AdaptedMarkerPlacementVerseOffsetPredictorFactory(AbstractVerseOffsetPredictorFactory):
    def create(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ) -> AdaptedMarkerPlacementVerseOffsetPredictor:
        return AdaptedMarkerPlacementVerseOffsetPredictor(
            source_tokens, target_tokens, source_verse_token_offsets, word_alignments
        )


class BreakScorer(ABC):
    def __init__(self, source_tokens: List[str], target_tokens: List[str], word_alignments: WordAlignments):
        self._source_tokens = source_tokens
        self._target_tokens = target_tokens
        self._word_alignments = word_alignments

    @abstractmethod
    def score_potential_break(
        self, src_break_offset: int, trg_break_offset: int, src_range: Tuple[int, int] | None = None
    ) -> float: ...


class ManualBreakScorer(BreakScorer):
    _MAX_WINDOW_SIZE = 5

    def score_potential_break(
        self, src_break_offset: int, trg_break_offset: int, src_range: Tuple[int, int] | None = None
    ) -> float:
        score = 0

        if src_range is not None:
            # favor splits that are closer to the middle of the range
            midpoint = (src_range[1] + src_range[0]) / 2
            score = ((midpoint - src_range[0]) - abs(src_break_offset - midpoint)) ** 2 * 0.0
        num_crossed_alignments = self._word_alignments.get_num_crossed_alignments(src_break_offset, trg_break_offset)
        score -= 20 * num_crossed_alignments

        num_alignments_in_windows: dict[int, int] = self._calculate_num_alignments_in_windows(
            src_break_offset, trg_break_offset
        )
        for distance_from_break, num_alignments_in_window in num_alignments_in_windows.items():
            score += distance_from_break * num_alignments_in_window

        # give a bonus for a split that is right next to punctuation
        if (
            src_break_offset > 0
            and not contains_letter(self._source_tokens[src_break_offset - 1])
            and trg_break_offset > 0
            and not contains_letter(self._target_tokens[trg_break_offset - 1])
        ):
            score += 5
        if (
            src_break_offset < len(self._source_tokens) - 1
            and not contains_letter(self._source_tokens[src_break_offset + 1])
            and trg_break_offset < len(self._target_tokens) - 1
            and not contains_letter(self._target_tokens[trg_break_offset + 1])
        ):
            score += 5

        # give a bonus for a split that is right next to capital letters
        if (
            src_break_offset < len(self._source_tokens) - 1
            and not contains_letter(self._source_tokens[src_break_offset + 1])
            and trg_break_offset < len(self._target_tokens) - 1
            and not contains_letter(self._target_tokens[trg_break_offset + 1])
        ):
            score += 5

        return score

    def _calculate_num_alignments_in_windows(self, src_break_offset: int, trg_break_offset: int) -> dict[int, int]:
        num_alignments_in_windows: dict[int, int] = {}
        for src_index in range(src_break_offset - self._MAX_WINDOW_SIZE, src_break_offset + self._MAX_WINDOW_SIZE + 1):
            if src_index < 0 or src_index >= len(self._source_tokens):
                continue
            aligned_trg_indices = self._word_alignments.get_target_aligned_words(src_index)
            for trg_index in aligned_trg_indices:
                if trg_break_offset - self._MAX_WINDOW_SIZE <= trg_index <= trg_break_offset + self._MAX_WINDOW_SIZE:
                    distance_from_break = abs(trg_index - trg_break_offset)
                    if distance_from_break in num_alignments_in_windows:
                        num_alignments_in_windows[distance_from_break] += 1
                    else:
                        num_alignments_in_windows[distance_from_break] = 1
        return num_alignments_in_windows


class AbstractBreakScorerFactory(ABC):
    @abstractmethod
    def create(
        self, source_tokens: List[str], target_tokens: List[str], word_alignments: WordAlignments
    ) -> BreakScorer: ...


class ManualBreakScorerFactory(AbstractBreakScorerFactory):
    def create(
        self, source_tokens: List[str], target_tokens: List[str], word_alignments: WordAlignments
    ) -> ManualBreakScorer:
        return ManualBreakScorer(source_tokens, target_tokens, word_alignments)


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
        sub_passages: Optional[List["SubPassage"]] = None,
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

        if sub_passages is not None:
            self._sub_passages = sub_passages
        else:
            self._sub_passages = []

    def segment_target_passage(
        self, verse_offset_predictor_factory: AbstractVerseOffsetPredictorFactory
    ) -> SegmentedPassage:
        verse_segmenter: VerseSegmenter = VerseSegmenter(
            verse_offset_predictor_factory,
            self._tokenized_source_text,
            self._tokenized_target_text,
            self._target_text,
            self._source_verse_token_offsets,
            self._word_alignments,
            self._sub_passages,
        )
        target_verses: List[Verse] = verse_segmenter.segment_verses(
            [verse.reference for verse in self._source_verses],
        )
        return SegmentedPassage(self._start_ref, self._end_ref, self._target_text, target_verses)

    def get_source_passage(self) -> SegmentedPassage:
        return SegmentedPassage(
            self._start_ref, self._end_ref, " ".join([v.text for v in self._source_verses]), self._source_verses
        )

    def to_json(self) -> Dict[str, Any]:

        return {
            "start_ref": str(self._start_ref),
            "end_ref": str(self._end_ref),
            "source_verses": [{"reference": str(v.reference), "text": v.text} for v in self._source_verses],
            "target_text": self._target_text,
            "word_alignments": self._word_alignments.to_json(),
            "tokenized_source_verses": [" ".join(tokens) for tokens in self._tokenized_source_verses],
            "tokenized_source_text": " ".join(self._tokenized_source_text),
            "tokenized_target_text": " ".join(self._tokenized_target_text),
            "verse_token_offsets": self._source_verse_token_offsets,
            "sub_passages": [sub_passage.to_json() for sub_passage in self._sub_passages],
        }

    @classmethod
    def from_json(cls, parallel_passage_json: Dict[str, Any]) -> "ParallelPassage":
        return cls(
            start_ref=VerseRef.from_string(parallel_passage_json["start_ref"]),
            end_ref=VerseRef.from_string(parallel_passage_json["end_ref"]),
            source_verses=[
                Verse(VerseRef.from_string(v["reference"]), v["text"]) for v in parallel_passage_json["source_verses"]
            ],
            target_text=parallel_passage_json["target_text"],
            word_alignments=WordAlignments.from_json(parallel_passage_json["word_alignments"]),
            tokenized_source_verses=[v.split() for v in parallel_passage_json.get("tokenized_source_verses", [])],
            tokenized_source_text=parallel_passage_json.get("tokenized_source_text", "").split(),
            tokenized_target_text=parallel_passage_json.get("tokenized_target_text", "").split(),
            verse_token_offsets=parallel_passage_json.get("verse_token_offsets", []),
            sub_passages=[SubPassage.from_json(sp) for sp in parallel_passage_json.get("sub_passages", [])],
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
        self._sub_passages: Optional[List["SubPassage"]] = None

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

    def split_into_subpassages(
        self, passage_splitter_factory: "PassageSplitterFactory", break_scorer_factory: AbstractBreakScorerFactory
    ) -> None:
        passage_splitter: "PassageSplitter" = passage_splitter_factory.create(
            self._tokenized_source_text,
            self._tokenized_target_text,
            self._verse_token_offsets,
            self._word_alignments,
            break_scorer_factory,
        )
        self._sub_passages = passage_splitter.split_into_minimal_segments()

    def get_sub_passages(self) -> List["SubPassage"]:
        if self._sub_passages is None:
            return [SubPassage(self._verse_token_offsets, self._tokenized_source_text, self._tokenized_target_text)]
        return self._sub_passages

    def build(self) -> ParallelPassage:
        if self._word_alignments is None:
            raise ValueError("Word alignments not loaded")
        if (
            self._sub_passages is not None
            and len(self._sub_passages) > 0
            and any(sub_passage.word_alignments is None for sub_passage in self._sub_passages)
        ):
            raise ValueError("Sub-passage word alignments not loaded")
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
            self._sub_passages,
        )


class ParallelPassageCollectionFactory(ABC):
    @abstractmethod
    def create(self, source_project_name: str, target_passage_file: Path) -> "ParallelPassageCollection": ...


class ParatextParallelPassageCollectionFactory(ParallelPassageCollectionFactory):
    def __init__(
        self,
        save_alignments: bool = False,
        subdivide_passages: bool = False,
        break_scorer_factory: AbstractBreakScorerFactory = ManualBreakScorerFactory(),
    ):
        self._save_alignments = save_alignments
        self._subdivide_passages = subdivide_passages
        self._break_scorer_factory = break_scorer_factory

    def create(self, source_project_name: str, target_passage_file: Path) -> "ParallelPassageCollection":
        target_passages = PassageReader(target_passage_file).get_passages()
        parallel_passage_builders = self._collect_parallel_passages(source_project_name, target_passages)
        alignment_generator = FastAlignConstrainedEflomalAlignmentGenerator(self._save_alignments, target_passage_file)
        self._create_word_alignment_matrix(parallel_passage_builders, alignment_generator)

        if self._subdivide_passages:
            self._split_into_subpassages(parallel_passage_builders, alignment_generator)

        parallel_passages = [parallel_passage_builder.build() for parallel_passage_builder in parallel_passage_builders]
        parallel_passage_collection = ParallelPassageCollection(parallel_passages)

        if self._save_alignments:
            with open(target_passage_file.with_suffix(".saved.json"), "w", encoding="utf-8") as f:
                json.dump(parallel_passage_collection.to_json(), f, ensure_ascii=False, indent=2)
        return parallel_passage_collection

    def _collect_parallel_passages(
        self, source_project_name: str, target_passages: List[Passage]
    ) -> List[ParallelPassageBuilder]:
        paratext_project_reader = ParatextProjectReader(source_project_name)
        parallel_passage_builders = [ParallelPassageBuilder(t.start_ref, t.end_ref, t.text) for t in target_passages]
        return paratext_project_reader.collect_verses(parallel_passage_builders)

    def _create_word_alignment_matrix(
        self, parallel_passage_builders: List[ParallelPassageBuilder], alignment_generator: "AlignmentGenerator"
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

    def _split_into_subpassages(
        self, parallel_passage_builders: List[ParallelPassageBuilder], alignment_generator: "AlignmentGenerator"
    ) -> None:
        passage_splitter_factory = PassageSplitterFactory()
        sub_passages: List[SubPassage] = []

        for parallel_passage in parallel_passage_builders:
            parallel_passage.split_into_subpassages(passage_splitter_factory, self._break_scorer_factory)
            sub_passages.extend(parallel_passage.get_sub_passages())

        for index, word_alignments in enumerate(
            alignment_generator.generate(
                [sub_passage.get_token_separated_source_text_for_alignment() for sub_passage in sub_passages],
                [sub_passage.get_token_separated_target_text_for_alignment() for sub_passage in sub_passages],
            )
        ):
            sub_passages[index].word_alignments = word_alignments


class SavedParallelPassageCollectionFactory(ParallelPassageCollectionFactory):

    def create(self, source_project_name: str, target_passage_file: Path) -> "ParallelPassageCollection":
        with open(target_passage_file.with_suffix(".saved.json"), "r", encoding="utf-8") as f:
            saved_passages_json = json.load(f)
            saved_passages = [ParallelPassage.from_json(p) for p in saved_passages_json]
            return ParallelPassageCollection(saved_passages)


@dataclass
class SubPassage:
    source_verse_token_offsets: List[int]
    source_tokens: List[str]
    target_tokens: List[str]
    word_alignments: Optional[WordAlignments] = None

    def get_token_separated_source_text_for_alignment(self) -> str:
        return " ".join(self.source_tokens)

    def get_token_separated_target_text_for_alignment(self) -> str:
        return " ".join(self.target_tokens)

    def to_json(self) -> Dict[str, Any]:
        return {
            "source_verse_token_offsets": self.source_verse_token_offsets,
            "source_tokens": " ".join(self.source_tokens),
            "target_tokens": " ".join(self.target_tokens),
            "word_alignments": self.word_alignments.to_json() if self.word_alignments is not None else None,
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "SubPassage":
        return SubPassage(
            source_verse_token_offsets=data["source_verse_token_offsets"],
            source_tokens=data["source_tokens"].split(),
            target_tokens=data["target_tokens"].split(),
            word_alignments=(
                WordAlignments.from_json(data["word_alignments"]) if data["word_alignments"] is not None else None
            ),
        )


class PassageSplitter:
    _MAX_WINDOW_SIZE = 5

    def __init__(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
        break_scorer: BreakScorer,
    ):
        self._source_tokens = source_tokens
        self._target_tokens = target_tokens
        self._source_verse_token_offsets = source_verse_token_offsets
        self._word_alignments = word_alignments
        self._break_scorer = break_scorer

    def split_into_minimal_segments(self) -> List[SubPassage]:
        src_to_trg_length_ratio = len(self._source_tokens) / len(self._target_tokens)

        subsegments: List[SubPassage] = []

        previous_src_break_offset = 0
        previous_trg_break_offset = 0

        # try to find smaller segments to break into
        for break_index, src_break_offset in enumerate(self._source_verse_token_offsets):
            src_break_offset_search_start = (
                self._source_verse_token_offsets[break_index - 1] + 1 if break_index > 0 else 1
            )

            best_src_split_index = -1
            best_trg_split_index = -1
            max_score = -1_000_000
            for potential_src_split_index in range(src_break_offset_search_start + 1, src_break_offset - 1):
                for potential_trg_split_index in range(
                    max(int(0.75 * src_to_trg_length_ratio * src_break_offset_search_start), previous_trg_break_offset),
                    min(len(self._target_tokens) + 1, int(1.25 * src_to_trg_length_ratio * src_break_offset)),
                ):
                    score = self._break_scorer.score_potential_break(
                        potential_src_split_index,
                        potential_trg_split_index,
                        (src_break_offset_search_start, src_break_offset),
                    )
                    if score > max_score:
                        max_score = score
                        best_src_split_index = potential_src_split_index
                        best_trg_split_index = potential_trg_split_index

            if max_score > 15:
                subsegments.append(
                    SubPassage(
                        source_verse_token_offsets=[
                            source_verse_token_offset - previous_src_break_offset
                            for source_verse_token_offset in self._source_verse_token_offsets
                            if previous_src_break_offset <= source_verse_token_offset < best_src_split_index
                        ],
                        source_tokens=self._source_tokens[previous_src_break_offset:best_src_split_index],
                        target_tokens=self._target_tokens[previous_trg_break_offset:best_trg_split_index],
                    )
                )
                print(" ".join(self._source_tokens[previous_src_break_offset:best_src_split_index]))
                print(" ".join(self._target_tokens[previous_trg_break_offset:best_trg_split_index]))
                print("---")
                previous_trg_break_offset = best_trg_split_index
                previous_src_break_offset = best_src_split_index
        subsegments.append(
            SubPassage(
                source_verse_token_offsets=[
                    source_verse_token_offset - previous_src_break_offset
                    for source_verse_token_offset in self._source_verse_token_offsets
                    if previous_src_break_offset <= source_verse_token_offset
                ],
                source_tokens=self._source_tokens[previous_src_break_offset:],
                target_tokens=self._target_tokens[previous_trg_break_offset:],
            )
        )
        return subsegments


class PassageSplitterFactory:
    def create(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
        break_scorer_factory: AbstractBreakScorerFactory,
    ) -> PassageSplitter:
        return PassageSplitter(
            source_tokens,
            target_tokens,
            source_verse_token_offsets,
            word_alignments,
            break_scorer_factory.create(source_tokens, target_tokens, word_alignments),
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
            compute_alignment_scores(
                Path(td, "src_align.txt"), Path(td, "trg_align.txt"), "eflomal", align_path, "grow-diag-final-and"
            )

            for src_passage, trg_passage, alignment_line in zip(
                source_passages, target_passages, load_corpus(align_path)
            ):
                currentWordAlignments: WordAlignmentsBuilder = WordAlignmentsBuilder(
                    len(src_passage.split()), len(trg_passage.split())
                )
                currentWordAlignments.add_aligments(AlignedWordPair.from_string(alignment_line))
                yield currentWordAlignments.build()

            if self._save_alignments:
                assert self._target_passage_file is not None
                saved_alignments_file = self._target_passage_file.with_suffix(".alignments.txt")
                shutil.copy(align_path, saved_alignments_file)

                src_tokenized_file = self._target_passage_file.with_suffix(".src.tokenized.txt")
                shutil.copy(src_path, src_tokenized_file)

                trg_tokenized_file = self._target_passage_file.with_suffix(".trg.tokenized.txt")
                shutil.copy(trg_path, trg_tokenized_file)


class FastAlignAlignmentGenerator(AlignmentGenerator):
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
            compute_alignment_scores(
                Path(td, "src_align.txt"), Path(td, "trg_align.txt"), "fast_align", align_path, "grow-diag-final-and"
            )

            for src_passage, trg_passage, alignment_line in zip(
                source_passages, target_passages, load_corpus(align_path)
            ):
                currentWordAlignments: WordAlignmentsBuilder = WordAlignmentsBuilder(
                    len(src_passage.split()), len(trg_passage.split())
                )
                currentWordAlignments.add_aligments(AlignedWordPair.from_string(alignment_line))
                yield currentWordAlignments.build()

            if self._save_alignments:
                assert self._target_passage_file is not None
                saved_alignments_file = self._target_passage_file.with_suffix(".alignments.txt")
                shutil.copy(align_path, saved_alignments_file)

                src_tokenized_file = self._target_passage_file.with_suffix(".src.tokenized.txt")
                shutil.copy(src_path, src_tokenized_file)

                trg_tokenized_file = self._target_passage_file.with_suffix(".trg.tokenized.txt")
                shutil.copy(trg_path, trg_tokenized_file)


class FastAlignConstrainedEflomalAlignmentGenerator(AlignmentGenerator):
    _MAX_CROSSINGS_FOR_FAST_ALIGN = 15

    def __init__(self, save_alignments: bool = False, target_passage_file: Optional[Path] = None):
        self._save_alignments = save_alignments
        self._target_passage_file = target_passage_file

    def generate(self, source_passages: List[str], target_passages: List[str]) -> Generator[WordAlignments, None, None]:

        fast_align_generator = FastAlignAlignmentGenerator(False, self._target_passage_file)
        eflomal_generator = EflomalAlignmentGenerator(False, self._target_passage_file)

        fast_align_alignments = list(fast_align_generator.generate(source_passages, target_passages))
        eflomal_alignments = list(eflomal_generator.generate(source_passages, target_passages))
        for fast_align_row, eflomal_row in zip(fast_align_alignments, eflomal_alignments):
            yield eflomal_row.remove_links_crossing_n(self._MAX_CROSSINGS_FOR_FAST_ALIGN, fast_align_row)


class SavedAlignmentGenerator(AlignmentGenerator):
    def __init__(self, src_tokenized_file: Path, trg_tokenized_file: Path, alignment_file: Path):
        self._src_tokenized_file = src_tokenized_file
        self._trg_tokenized_file = trg_tokenized_file
        self._alignment_file = alignment_file

    def generate(self, source_passages: List[str], target_passages: List[str]) -> Generator[WordAlignments, None, None]:
        for src_passage, trg_passage, alignment_line in zip(
            load_corpus(self._src_tokenized_file),
            load_corpus(self._trg_tokenized_file),
            load_corpus(self._alignment_file),
        ):
            currentWordAlignments: WordAlignmentsBuilder = WordAlignmentsBuilder(
                len(src_passage.split()), len(trg_passage.split())
            )
            currentWordAlignments.add_aligments(AlignedWordPair.from_string(alignment_line))
            yield currentWordAlignments.build()


VerseCollectorType = TypeVar("VerseCollectorType", bound=VerseCollector)


class ParatextProjectReader:
    def __init__(self, project_name: str):
        self._project_name = project_name

    def collect_verses(self, verse_collectors: List[VerseCollectorType]) -> List[VerseCollectorType]:
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

    def _get_all_required_books(self, verse_collectors: List[VerseCollectorType]) -> Set[str]:
        all_books: Set[str] = set()
        for verse_collector in verse_collectors:
            all_books.add(verse_collector.start_ref.book)
            all_books.add(verse_collector.end_ref.book)
        return all_books


class ParallelPassageCollection:
    def __init__(self, parallel_passages: List[ParallelPassage]):
        self._parallel_passages = parallel_passages

    def segment_target_passages(
        self, verse_offset_predictor_factory: AbstractVerseOffsetPredictorFactory
    ) -> Iterable[SegmentedPassage]:
        for passage in self._parallel_passages:
            yield passage.segment_target_passage(verse_offset_predictor_factory)

    def get_source_segmented_passages(self) -> List[SegmentedPassage]:
        return [parallel_passage.get_source_passage() for parallel_passage in self._parallel_passages]

    def to_json(self) -> List[Dict[str, Any]]:
        return [parallel_passage.to_json() for parallel_passage in self._parallel_passages]


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
        "--use-saved-alignments",
        help="Use pre-computed alignments from a previous run",
        default=None,
        action="store_true",
    )
    parser.add_argument("--vref", help="Output vref file for target verses", default=None, action="store_true")
    args = parser.parse_args()

    if args.use_saved_alignments:
        parallel_passages = SavedParallelPassageCollectionFactory().create(
            args.source_project, Path(args.target_passages)
        )
    else:
        parallel_passages = ParatextParallelPassageCollectionFactory(
            args.save_alignments, subdivide_passages=True
        ).create(args.source_project, Path(args.target_passages))
    src_segmented_passages = parallel_passages.get_source_segmented_passages()

    verse_offset_predictor_factory = ScoringFunctionVerseOffsetPredictorFactory(
        break_scorer_factory=ManualBreakScorerFactory()
    )
    trg_segmented_passages = list(parallel_passages.segment_target_passages(verse_offset_predictor_factory))

    src_path = Path(args.target_passages).with_suffix(".src.txt")
    trg_path = Path(args.target_passages).with_suffix(".trg.txt")
    with open(src_path, "w", encoding="utf-8") as src_output, open(trg_path, "w", encoding="utf-8") as trg_output:
        for src_passage, trg_passage in zip(src_segmented_passages, trg_segmented_passages):
            src_passage.write_to_file(src_output)
            trg_passage.write_to_file(trg_output)

    if args.vref is not None:
        vref_path = Path(args.target_passages).with_suffix(".vref.txt")
        template_vref_path = SIL_NLP_ENV.assets_dir / "vref.txt"

        verse_map: Dict[str, str] = {
            str(verse.reference.to_versification(ORIGINAL_VERSIFICATION)): verse.text
            for trg_passage in trg_segmented_passages
            for verse in trg_passage.verses
        }

        with (
            open(template_vref_path, "r", encoding="utf-8") as template_file,
            open(vref_path, "w", encoding="utf-8") as vref_output,
        ):
            for line in template_file:
                template_ref = line.rstrip("\n")
                vref_output.write(f"{verse_map.get(template_ref, '')}\n")

    if args.compare_against is not None:
        reference_segmentations = ReferenceVerseSegmentationReader().read_passages(
            args.compare_against, Path(args.target_passages)
        )

        trg_gold_path = Path(args.target_passages).with_suffix(".trg.gold.txt")
        with open(trg_gold_path, "w", encoding="utf-8") as trg_gold_output:
            for reference_passage in reference_segmentations:
                reference_passage.write_to_file(trg_gold_output)

        segmentation_evaluator = SegmentationEvaluator(reference_segmentations)
        segmentation_evaluation = segmentation_evaluator.evaluate_segmentation(trg_segmented_passages)
        print(
            f"Segmentation accuracy = {segmentation_evaluation.get_accuracy() * 100}%, average distance = {segmentation_evaluation.get_average_distance()} characters"
        )


if __name__ == "__main__":
    main()
