from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import regex
from machine.scripture import VerseRef

from .passage import Verse
from .verse_offset_predictors import AbstractVerseOffsetPredictorFactory
from .word_alignments import WordAlignments

if TYPE_CHECKING:
    from .sub_passage import SubPassage


class VerseSegmenter:
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
                sub_passage_word_alignments = sub_passage.word_alignments
                if sub_passage_word_alignments is None:
                    continue
                for sub_passage_predicted_offset in self._verse_offset_predictor_factory.create(
                    sub_passage.source_tokens,
                    sub_passage.target_tokens,
                    sub_passage.source_verse_token_offsets,
                    sub_passage_word_alignments,
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
