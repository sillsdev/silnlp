import argparse
import logging
from enum import Enum
from pathlib import Path
from typing import (
    Collection,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import regex
from machine.corpora import (
    UsfmAttribute,
    UsfmParserHandler,
    UsfmParserState,
    parse_usfm,
)
from machine.scripture import get_chapters

from silnlp.nmt.config import Config, DataFileType

from ..nmt.config_utils import load_config
from .environment import SIL_NLP_ENV
from .paratext import parse_books

LOGGER = logging.getLogger(__package__ + ".analyze_quote_convention")


class QuotationDirection(Enum):
    Opening = "Opening"
    Closing = "Closing"


class SingleLevelQuoteConvention:
    def __init__(self, opening_quote: str, closing_quote: str):
        self.opening_quote = opening_quote
        self.closing_quote = closing_quote

    def get_opening_quote(self) -> str:
        return self.opening_quote

    def get_closing_quote(self) -> str:
        return self.closing_quote


class QuoteConvention:
    def __init__(self, name: str, levels: list[SingleLevelQuoteConvention]):
        self.name = name
        self.levels = levels

    def get_name(self) -> str:
        return self.name

    def num_levels(self) -> int:
        return len(self.levels)

    def get_level_convention(self, level: int) -> SingleLevelQuoteConvention:
        return self.levels[level - 1]

    def get_expected_quotation_mark(self, depth: int, direction: QuotationDirection) -> str:
        if depth > len(self.levels):
            return ""
        return (
            self.get_level_convention(depth).get_opening_quote()
            if direction == QuotationDirection.Opening
            else self.get_level_convention(depth).get_closing_quote()
        )

    def includes_opening_quotation_mark(self, opening_quotation_mark: str) -> bool:
        for level in self.levels:
            if level.get_opening_quote() == opening_quotation_mark:
                return True
        return False

    def includes_closing_quotation_mark(self, closing_quotation_mark: str) -> bool:
        for level in self.levels:
            if level.get_closing_quote() == closing_quotation_mark:
                return True
        return False

    def is_compatible_with_observed_quotation_marks(
        self, opening_quotation_marks: list[str], closing_quotation_marks: list[str]
    ) -> bool:
        for opening_quotation_mark in opening_quotation_marks:
            if not self.includes_opening_quotation_mark(opening_quotation_mark):
                return False
        for closing_quotation_mark in closing_quotation_marks:
            if not self.includes_closing_quotation_mark(closing_quotation_mark):
                return False

        # we require the first-level quotes to have been observed
        if self.get_level_convention(1).get_opening_quote() not in opening_quotation_marks:
            return False
        if self.get_level_convention(1).get_closing_quote() not in closing_quotation_marks:
            return False
        return True

    def print_summary(self) -> None:
        print(self.get_name())
        for level, convention in enumerate(self.levels):
            ordinal_name = self._get_ordinal_name(level + 1)
            print("%s%s-level quote%s" % (convention.get_opening_quote(), ordinal_name, convention.get_closing_quote()))

    def _get_ordinal_name(self, level) -> str:
        if level == 1:
            return "First"
        if level == 2:
            return "Second"
        if level == 3:
            return "Third"
        if level == 4:
            return "Fourth"
        return str(level) + "th"


class QuoteConventionSet:
    def __init__(self, conventions: List[QuoteConvention]):
        self.conventions = conventions
        self._create_quotation_mark_pair_map()

    def _create_quotation_mark_pair_map(self) -> None:
        self.quotation_mark_pairs: dict[str, set[str]] = dict()
        for convention in self.conventions:
            for level in range(0, convention.num_levels()):
                opening_quote = convention.get_level_convention(level).get_opening_quote()
                closing_quote = convention.get_level_convention(level).get_closing_quote()
                if opening_quote not in self.quotation_mark_pairs:
                    self.quotation_mark_pairs[opening_quote] = set()
                self.quotation_mark_pairs[opening_quote].add(closing_quote)

    def get_possible_opening_marks(self) -> list[str]:
        return list(self.quotation_mark_pairs.keys())

    def get_possible_closing_marks(self) -> list[str]:
        return [
            closing_mark for closing_mark_set in self.quotation_mark_pairs.values() for closing_mark in closing_mark_set
        ]

    def is_valid_opening_quotation_mark(self, quotation_mark: str) -> None:
        return quotation_mark in self.quotation_mark_pairs

    def is_valid_closing_quotation_mark(self, quotation_mark: str) -> None:
        for closing_mark_set in self.quotation_mark_pairs.values():
            if quotation_mark in closing_mark_set:
                return True
        return False

    def are_marks_a_valid_pair(self, opening_mark: str, closing_mark: str) -> bool:
        return (opening_mark in self.quotation_mark_pairs) and (closing_mark in self.quotation_mark_pairs[opening_mark])

    def filter_to_compatible_quote_conventions(
        self, opening_quotation_marks: list[str], closing_quotation_marks: list[str]
    ) -> "QuoteConventionSet":
        return QuoteConventionSet(
            [
                convention
                for convention in self.conventions
                if convention.is_compatible_with_observed_quotation_marks(
                    opening_quotation_marks, closing_quotation_marks
                )
            ]
        )

    def find_most_similar_convention(
        self, tabulated_quotation_marks: "QuotationMarkTabulator"
    ) -> Tuple[QuoteConvention, float]:
        best_similarity: float = float("-inf")
        best_quote_convention: QuoteConvention | None = None
        for quote_convention in self.conventions:
            similarity = tabulated_quotation_marks.calculate_similarity(quote_convention)
            if similarity > best_similarity:
                best_similarity = similarity
                best_quote_convention = quote_convention

        return (best_quote_convention, best_similarity)

    def print_summary(self) -> None:
        print("Opening quotation marks must be one of the following: ", self.get_possible_opening_marks())
        print("Closing quotation marks must be one of the following: ", self.get_possible_closing_marks())


class UsfmMarkerType(Enum):
    ParagraphMarker = "ParagraphMarker"
    CharacterMarker = "CharacterMarker"
    VerseMarker = "VerseMarker"
    ChapterMarker = "ChapterMarker"
    EmbedMarker = "Embed"
    Other = "Other"
    NoMarker = "NoMarker"


class TextSegment:
    def __init__(self):
        self.text = ""
        self.immediate_preceding_marker: UsfmMarkerType = UsfmMarkerType.NoMarker
        self.markers_in_preceding_context: Set[UsfmMarkerType] = set()
        self.previous_segment: TextSegment | None = None
        self.next_segment: TextSegment | None = None
        self.index_in_verse: int = 0
        self.num_segments_in_verse: int = 0

    def get_text(self) -> str:
        return self.text

    def length(self) -> int:
        return len(self.text)

    def substring_before(self, index: int) -> str:
        return self.text[0:index]

    def substring_after(self, index: int) -> str:
        return self.text[index:-1]

    def get_immediate_preceding_context(self) -> UsfmMarkerType:
        return self.immediate_preceding_marker

    def is_marker_in_preceding_context(self, marker: UsfmMarkerType) -> bool:
        return marker in self.markers_in_preceding_context

    def get_previous_segment(self) -> "TextSegment | None":
        return self.previous_segment

    def get_next_segment(self) -> "TextSegment | None":
        return self.next_segment

    def is_first_segment_in_verse(self) -> bool:
        return self.index_in_verse == 0

    def is_last_segment_in_verse(self) -> bool:
        return self.index_in_verse == self.num_segments_in_verse - 1

    # These setters need to be done outside the builder to avoid circular dependencies
    def set_next_segment(self, next_segment: "TextSegment") -> None:
        self.next_segment = next_segment

    def set_index_in_verse(self, index_in_verse: int) -> None:
        self.index_in_verse = index_in_verse

    def set_num_segments_in_verse(self, num_segments_in_verse: int) -> None:
        self.num_segments_in_verse = num_segments_in_verse

    class Builder:
        def __init__(builder_self):
            builder_self.text_segment = TextSegment()

        def set_previous_segment(builder_self, previous_segment: "TextSegment") -> "TextSegment.Builder":
            builder_self.text_segment.previous_segment = previous_segment
            return builder_self

        def set_next_segment(builder_self, next_segment: "TextSegment") -> "TextSegment.Builder":
            builder_self.text_segment.next_segment = next_segment
            return builder_self

        def add_preceding_marker(builder_self, marker: UsfmMarkerType) -> "TextSegment.Builder":
            builder_self.text_segment.immediate_preceding_marker = marker
            builder_self.text_segment.markers_in_preceding_context.add(marker)

        def set_text(builder_self, text: str) -> "TextSegment.Builder":
            builder_self.text_segment.text = text

        def build(builder_self) -> "TextSegment":
            return builder_self.text_segment


class Verse:
    def __init__(self, text_segments: list[TextSegment]):
        self.text_segments = text_segments
        self._index_text_segments()

    def _index_text_segments(self) -> None:
        for index, text_segment in enumerate(self.text_segments):
            text_segment.set_index_in_verse(index)
            text_segment.set_num_segments_in_verse(len(self.text_segments))

    def get_text_segments(self) -> list[TextSegment]:
        return self.text_segments


class Chapter:
    def __init__(self, verses: list[Verse]):
        self.verses = verses

    def get_verses(self) -> list[Verse]:
        return self.verses


class QuotationMarkMetadata:
    def __init__(
        self, quotation_mark: str, depth: int, direction: QuotationDirection, start_index: int, end_index: int
    ):
        self.quotation_mark = quotation_mark
        self.depth = depth
        self.direction = direction
        self.start_index = start_index
        self.end_index = end_index

    def get_quotation_mark(self) -> str:
        return self.quotation_mark

    def get_depth(self) -> int:
        return self.depth

    def get_direction(self) -> QuotationDirection:
        return self.direction

    def get_start_index(self) -> int:
        return self.start_index

    def get_end_index(self) -> int:
        return self.end_index


class QuotationMarkMatch:

    # extra stuff in the regex to handle Western Cham
    letter_pattern = regex.compile(r"[\p{L}\U0001E200-\U0001E28F]", regex.U)

    def __init__(self, text_segment: TextSegment, start_index: int, end_index: int):
        self.text_segment = text_segment
        self.start_index = start_index
        self.end_index = end_index

    def get_quotation_mark(self) -> str:
        return self.text_segment.get_text()[self.start_index : self.end_index]

    def quotation_mark_matches(self, regex_pattern: regex.Pattern) -> bool:
        return regex_pattern.search(self.get_quotation_mark())

    def next_character_matches(self, regex_pattern: regex.Pattern) -> bool:
        return self.get_next_character() is not None and regex_pattern.search(self.get_next_character())

    def previous_character_matches(self, regex_pattern: regex.Pattern) -> bool:
        return self.get_previous_character() is not None and regex_pattern.search(self.get_previous_character())

    def get_previous_character(self) -> str | None:
        if self.start_index == 0:
            if (
                self.text_segment.get_previous_segment() is not None
                and not self.text_segment.is_marker_in_preceding_context(UsfmMarkerType.ParagraphMarker)
            ):
                return self.text_segment.get_previous_segment().get_text()[-1]
            return ""
        return self.text_segment.get_text()[self.start_index - 1]

    def get_next_character(self) -> str | None:
        if self.end_index == len(self.text_segment.get_text()):
            if (
                self.text_segment.get_next_segment() is not None
                and not self.text_segment.get_next_segment().is_marker_in_preceding_context(
                    UsfmMarkerType.ParagraphMarker
                )
            ):
                return self.text_segment.get_next_segment().get_text()[0]
            return ""
        return self.text_segment.get_text()[self.end_index]

    # this assumes that the two matches occur in the same verse
    def precedes(self, other: "QuotationMarkMatch") -> bool:
        return self.text_segment.index_in_verse < other.text_segment.index_in_verse or (
            self.text_segment.index_in_verse == other.text_segment.index_in_verse
            and self.start_index < other.start_index
        )

    def get_text_segment(self) -> TextSegment:
        return self.text_segment

    def get_start_index(self) -> int:
        return self.start_index

    def get_end_index(self) -> int:
        return self.end_index

    def get_context(self) -> str:
        return self.text_segment.get_text()[
            max(self.start_index - 10, 0) : min(self.end_index + 10, len(self.text_segment.get_text()))
        ]

    def resolve(self, depth: int, direction: QuotationDirection) -> QuotationMarkMetadata:
        return QuotationMarkMetadata(self.get_quotation_mark(), depth, direction, self.start_index, self.end_index)

    def is_at_start_of_segment(self) -> bool:
        return self.start_index == 0

    def is_at_end_of_segment(self) -> bool:
        return self.end_index == self.text_segment.length()

    def has_leading_letters(self) -> bool:
        if self.letter_pattern.search(self.text_segment.substring_before(self.start_index)):
            return True
        return False

    def has_trailing_letters(self) -> bool:
        if self.letter_pattern.search(self.text_segment.substring_after(self.end_index)):
            return True
        return False


class PreliminaryQuotationAnalyzer:
    quote_pattern = regex.compile(r"(\p{Quotation_Mark}|<<|>>|<|>)", regex.U)
    apostrophe_characters = ["'"]
    apostrophe_pattern = regex.compile(r"[\'\u2019]", regex.U)
    whitespace_pattern = regex.compile(r"^[\s~]*$", regex.U)

    def __init__(self, quote_conventions: QuoteConventionSet):
        self.quote_conventions = quote_conventions
        self.reset_analysis()

    def reset_analysis(self) -> None:
        self.num_characters = 0
        self.num_apostrophes = 0
        self.word_initial_occurrences: dict[str, int] = dict()
        self.mid_word_occurrences: dict[str, int] = dict()
        self.word_final_occurrences: dict[str, int] = dict()
        self.verse_starting_quotation_mark_counts: dict[str, int] = dict()
        self.verse_ending_quotation_mark_counts: dict[str, int] = dict()
        self.earlier_quotation_mark_counts: dict[str, int] = dict()
        self.later_quotation_mark_counts: dict[str, int] = dict()

    def narrow_down_possible_quote_conventions(self, chapters: list[Chapter]) -> QuoteConventionSet:
        for chapter in chapters:
            self.analyze_quotation_marks_for_chapter(chapter)
        return self.select_compatible_quote_conventions()

    def analyze_quotation_marks_for_chapter(self, chapter: Chapter) -> None:
        for verse in chapter.get_verses():
            self.analyze_quotation_marks_for_verse(verse)

    def analyze_quotation_marks_for_verse(self, verse: Verse) -> None:
        self.count_characters_in_verse(verse)
        quotation_marks = QuotationMarkFinder(self.quote_conventions).find_all_potential_quotation_marks_in_verse(verse)
        self.analyze_quotation_mark_sequence(quotation_marks)
        self.count_verse_starting_and_ending_quotation_marks(quotation_marks)

    def analyze_quotation_mark_sequence(self, quotation_marks: list[QuotationMarkMatch]) -> None:
        grouped_quotation_marks: dict[str, list[QuotationMarkMatch]] = self.group_quotation_marks(quotation_marks)
        for mark1, matches1 in grouped_quotation_marks.items():
            # handle cases of identical opening/closing marks
            if len(matches1) == 2 and (
                mark1 == '"'
                or mark1 == "'"
                or (
                    mark1 == "\u201d"
                    and "\u201c" not in grouped_quotation_marks
                    and "\u201e" not in grouped_quotation_marks
                )
                or (
                    mark1 == "\u2019"
                    and "\u2018" not in grouped_quotation_marks
                    and "\u201a" not in grouped_quotation_marks
                )
                or (mark1 == "\u00bb" and "\u00ab" not in grouped_quotation_marks)
            ):
                if mark1 not in self.earlier_quotation_mark_counts:
                    self.earlier_quotation_mark_counts[mark1] = 0
                self.earlier_quotation_mark_counts[mark1] += 1
                if mark1 not in self.later_quotation_mark_counts:
                    self.later_quotation_mark_counts[mark1] = 0
                self.later_quotation_mark_counts[mark1] += 1
                continue
            # skip verses where quotation mark pairs are ambiguous
            if len(matches1) > 1:
                continue
            # find matching closing marks
            for mark2, matches2 in grouped_quotation_marks.items():
                if len(matches2) > 1 or not self.quote_conventions.are_marks_a_valid_pair(mark1, mark2):
                    continue
                if not matches1[0].precedes(matches2[0]):
                    continue
                self.record_quotation_mark_sequence(mark1, mark2)

    def group_quotation_marks(self, quotation_marks: list[QuotationMarkMatch]) -> dict[str, int]:
        grouped_quotation_marks: dict[str, int] = dict()
        for quotation_mark_match in quotation_marks:
            if quotation_mark_match.get_quotation_mark() not in grouped_quotation_marks:
                grouped_quotation_marks[quotation_mark_match.get_quotation_mark()] = []
            grouped_quotation_marks[quotation_mark_match.get_quotation_mark()].append(quotation_mark_match)
        return grouped_quotation_marks

    def record_quotation_mark_sequence(self, earlier_mark: str, later_mark: str) -> None:
        if earlier_mark not in self.earlier_quotation_mark_counts:
            self.earlier_quotation_mark_counts[earlier_mark] = 0
        self.earlier_quotation_mark_counts[earlier_mark] += 1
        if later_mark not in self.later_quotation_mark_counts:
            self.later_quotation_mark_counts[later_mark] = 0
        self.later_quotation_mark_counts[later_mark] += 1

    def count_verse_starting_and_ending_quotation_marks(self, quotation_marks: list[QuotationMarkMatch]) -> None:
        for quotation_mark_match in quotation_marks:
            if self.apostrophe_pattern.search(quotation_mark_match.get_quotation_mark()):
                self.count_apostrophe(quotation_mark_match)
            if (
                quotation_mark_match.get_text_segment().is_first_segment_in_verse()
                and not quotation_mark_match.has_leading_letters()
            ):
                self.process_verse_starting_quotation_mark(quotation_mark_match)
            if (
                quotation_mark_match.get_text_segment().is_last_segment_in_verse()
                and not quotation_mark_match.has_trailing_letters()
            ):
                self.process_verse_ending_quotation_mark(quotation_mark_match)

    def process_verse_starting_quotation_mark(self, quotation_mark_match: QuotationMarkMatch) -> None:
        if quotation_mark_match.get_quotation_mark() not in self.verse_starting_quotation_mark_counts:
            self.verse_starting_quotation_mark_counts[quotation_mark_match.get_quotation_mark()] = 0
        self.verse_starting_quotation_mark_counts[quotation_mark_match.get_quotation_mark()] += 1

    def process_verse_ending_quotation_mark(self, quotation_mark_match: QuotationMarkMatch) -> None:
        if quotation_mark_match.get_quotation_mark() not in self.verse_ending_quotation_mark_counts:
            self.verse_ending_quotation_mark_counts[quotation_mark_match.get_quotation_mark()] = 0
        self.verse_ending_quotation_mark_counts[quotation_mark_match.get_quotation_mark()] += 1

    def count_characters_in_verse(self, verse: Verse) -> None:
        for text_segment in verse.get_text_segments():
            self.count_characters_in_text_segment(text_segment)

    def count_characters_in_text_segment(self, text_segment: TextSegment) -> None:
        self.num_characters += len(text_segment.get_text())

    def count_apostrophe(self, apostrophe_match: QuotationMarkMatch) -> None:
        apostrophe: str = apostrophe_match.get_quotation_mark()
        if self.is_match_word_initial(apostrophe_match):
            self.count_word_initial_apostrophe(apostrophe)
        elif self.is_match_mid_word(apostrophe_match):
            self.count_mid_word_apostrophe(apostrophe)
        elif self.is_match_word_final(apostrophe_match):
            self.count_word_final_apostrophe(apostrophe)

    def is_match_word_initial(self, apostrophe_match: QuotationMarkMatch) -> bool:
        if apostrophe_match.next_character_matches(self.whitespace_pattern):
            return False
        if not apostrophe_match.is_at_start_of_segment() and not apostrophe_match.previous_character_matches(
            self.whitespace_pattern
        ):
            return False
        return True

    def count_word_initial_apostrophe(self, apostrophe: str) -> None:
        if apostrophe not in self.word_initial_occurrences:
            self.word_initial_occurrences[apostrophe] = 0
        self.word_initial_occurrences[apostrophe] += 1

    def is_match_mid_word(self, apostrophe_match: QuotationMarkMatch) -> bool:
        if apostrophe_match.next_character_matches(self.whitespace_pattern):
            return False
        if apostrophe_match.previous_character_matches(self.whitespace_pattern):
            return False
        return True

    def count_mid_word_apostrophe(self, apostrophe: str) -> None:
        if apostrophe not in self.mid_word_occurrences:
            self.mid_word_occurrences[apostrophe] = 0
        self.mid_word_occurrences[apostrophe] += 1

    def is_match_word_final(self, apostrophe_match: QuotationMarkMatch) -> bool:
        if not apostrophe_match.is_at_end_of_segment() and not apostrophe_match.next_character_matches(
            self.whitespace_pattern
        ):
            return False
        if apostrophe_match.previous_character_matches(self.whitespace_pattern):
            return False
        return True

    def count_word_final_apostrophe(self, apostrophe: str) -> None:
        if apostrophe not in self.word_final_occurrences:
            self.word_final_occurrences[apostrophe] = 0
        self.word_final_occurrences[apostrophe] += 1

    def select_compatible_quote_conventions(self) -> QuoteConventionSet:
        opening_quotation_marks = self.find_opening_quotation_marks()
        closing_quotation_marks = self.find_closing_quotation_marks()

        return self.quote_conventions.filter_to_compatible_quote_conventions(
            opening_quotation_marks, closing_quotation_marks
        )

    def find_opening_quotation_marks(self) -> None:
        return [
            quotation_mark
            for quotation_mark in self.quote_conventions.get_possible_opening_marks()
            if self.is_opening_quotation_mark(quotation_mark)
        ]

    def is_opening_quotation_mark(self, quotation_mark: str) -> bool:
        if self.is_apostrophe_only(quotation_mark):
            return False

        num_early_occurrences: int = self.count_early_occurrences(quotation_mark)
        num_late_occurrences: int = self.count_late_occurrences(quotation_mark)

        if (
            num_late_occurrences == 0 and num_early_occurrences > 5
        ) or num_early_occurrences > num_late_occurrences * 10:
            return True
        if (
            num_early_occurrences > 5
            and abs(num_late_occurrences - num_early_occurrences) / num_early_occurrences < 0.2
            and quotation_mark in ['"', "'", "\u201d", "\u2019", "\u00bb"]
        ):
            return True
        return False

    def find_closing_quotation_marks(self) -> None:
        return [
            quotation_mark
            for quotation_mark in self.quote_conventions.get_possible_closing_marks()
            if self.is_closing_quotation_mark(quotation_mark)
        ]

    def is_closing_quotation_mark(self, quotation_mark: str) -> bool:
        if self.is_apostrophe_only(quotation_mark):
            return False

        num_early_occurrences: int = self.count_early_occurrences(quotation_mark)
        num_late_occurrences: int = self.count_late_occurrences(quotation_mark)
        if (
            num_early_occurrences == 0 and num_late_occurrences > 5
        ) or num_late_occurrences > num_early_occurrences * 10:
            return True
        if (
            num_late_occurrences > 5
            and abs(num_late_occurrences - num_early_occurrences) / num_late_occurrences < 0.2
            and quotation_mark in ['"', "'", "\u201d", "\u2019", "\u00bb"]
        ):
            return True
        return False

    def count_early_occurrences(self, quotation_mark: str) -> int:
        # num_verse_starting_occurrences: int = (
        #    self.verse_starting_quotation_mark_counts[quotation_mark]
        #    if quotation_mark in self.verse_starting_quotation_mark_counts
        #    else 0
        # )
        num_early_paired_occurrences: int = (
            self.earlier_quotation_mark_counts[quotation_mark]
            if quotation_mark in self.earlier_quotation_mark_counts
            else 0
        )

        # return num_verse_starting_occurrences + num_early_paired_occurrences
        return num_early_paired_occurrences

    def count_late_occurrences(self, quotation_mark: str) -> int:
        # num_verse_ending_occurrences: int = (
        #    self.verse_ending_quotation_mark_counts[quotation_mark]
        #    if quotation_mark in self.verse_ending_quotation_mark_counts
        #    else 0
        # )
        num_late_paired_occurrences: int = (
            self.later_quotation_mark_counts[quotation_mark]
            if quotation_mark in self.later_quotation_mark_counts
            else 0
        )

        # return num_verse_ending_occurrences + num_late_paired_occurrences
        return num_late_paired_occurrences

    def is_apostrophe_only(self, mark: str) -> bool:
        if not self.apostrophe_pattern.search(mark):
            return False

        num_initial_apostrophes = self.word_initial_occurrences[mark] if mark in self.word_initial_occurrences else 0
        num_mid_apostrophes = self.mid_word_occurrences[mark] if mark in self.mid_word_occurrences else 0
        num_final_apostrophes = self.word_final_occurrences[mark] if mark in self.word_final_occurrences else 0
        num_total_apostrophes = num_initial_apostrophes + num_mid_apostrophes + num_final_apostrophes

        if num_total_apostrophes > 0 and (
            num_initial_apostrophes / num_total_apostrophes < 0.1 or num_final_apostrophes / num_total_apostrophes < 0.1
        ):
            return True

        if (
            num_total_apostrophes > 0
            and abs(num_initial_apostrophes - num_final_apostrophes) / num_total_apostrophes > 0.3
            and num_mid_apostrophes / num_total_apostrophes > 0.3
        ):
            return True

        if num_total_apostrophes > self.num_characters / 50:
            return True

        return False


class QuotationMarkFinder:
    quote_pattern = regex.compile(r"(\p{Quotation_Mark}|<<|>>|<|>)", regex.U)

    def __init__(self, quote_convention_set: QuoteConventionSet):
        self.quote_convention_set = quote_convention_set

    def find_all_potential_quotation_marks_in_chapter(self, chapter: Chapter) -> list[QuotationMarkMatch]:
        quotation_matches: list[QuotationMarkMatch] = []
        for verse in chapter.get_verses():
            quotation_matches.extend(self.find_all_potential_quotation_marks_in_verse(verse))
        return quotation_matches

    def find_all_potential_quotation_marks_in_verse(self, verse: Verse) -> list[QuotationMarkMatch]:
        quotation_matches: list[QuotationMarkMatch] = []
        for text_segment in verse.get_text_segments():
            quotation_matches.extend(self.find_all_potential_quotation_marks_in_text_segment(text_segment))
        return quotation_matches

    def find_all_potential_quotation_marks_in_text_segment(self, text_segment: TextSegment) -> list[QuotationMarkMatch]:
        quotation_matches: list[QuotationMarkMatch] = []
        for quote_match in self.quote_pattern.finditer(text_segment.get_text()):
            if self.quote_convention_set.is_valid_opening_quotation_mark(
                quote_match.group()
            ) or self.quote_convention_set.is_valid_closing_quotation_mark(quote_match.group()):
                quotation_matches.append(QuotationMarkMatch(text_segment, quote_match.start(), quote_match.end()))
        return quotation_matches


class QuotationMarkResolver:
    quote_pattern = regex.compile(r"(?<=(.)|^)(\p{Quotation_Mark}|<<|>>|<|>)(?=(.)|$)", regex.U)
    apostrophe_pattern = regex.compile(r"[\'\u2019\u2018]", regex.U)
    whitespace_pattern = regex.compile(r"^[\s~]*$", regex.U)
    latin_letter_pattern = regex.compile(r"^\p{script=Latin}$", regex.U)
    punctuation_pattern = regex.compile(r"^[\.,;\?!\)\]\-—۔،؛]$", regex.U)

    def __init__(self, quote_convention_set: QuoteConventionSet):
        self.quote_convention_set = quote_convention_set
        self.quote_matches: list[QuotationMarkMatch] = []
        self.quotation_stack: list[QuotationMarkMetadata] = []
        self.quotation_continuer_stack: list[QuotationMarkMetadata] = []
        self.current_depth: int = 0

    def is_quotation_continuer(
        self,
        quote_match: QuotationMarkMatch,
        previous_match: QuotationMarkMatch | None,
        next_match: QuotationMarkMatch | None,
    ) -> None:
        if not quote_match.get_text_segment().is_marker_in_preceding_context(UsfmMarkerType.ParagraphMarker):
            return False
        if self.current_depth == 0:
            return False

        if len(self.quotation_continuer_stack) == 0:
            if quote_match.start_index > 0:
                return False
            if (
                quote_match.get_quotation_mark()
                != self.quotation_stack[len(self.quotation_continuer_stack) - 1].get_quotation_mark()
            ):
                return False
            if len(self.quotation_stack) > 1:
                if next_match is None or next_match.get_start_index() != quote_match.get_end_index():
                    return False
        else:
            if quote_match.get_quotation_mark() != self.quotation_continuer_stack[-1].get_quotation_mark():
                return False

        return True

    def is_expecting_quotation_continuer(self) -> bool:
        return len(self.quotation_continuer_stack) > 0 and len(self.quotation_continuer_stack) < len(
            self.quotation_stack
        )

    def resolve_quotation_marks(
        self, quote_matches: list[QuotationMarkMatch]
    ) -> Generator[QuotationMarkMetadata, None, None]:
        for quote_index, quote_match in enumerate(quote_matches):
            previous_mark = None if quote_index == 0 else quote_matches[quote_index - 1]
            next_mark = None if quote_index == len(quote_matches) - 1 else quote_matches[quote_index + 1]
            yield from self.resolve_quotation_mark(quote_match, previous_mark, next_mark)

    def resolve_quotation_mark(
        self,
        quote_match: QuotationMarkMatch,
        previous_mark: QuotationMarkMatch | None,
        next_mark: QuotationMarkMatch | None,
    ) -> Generator[QuotationMarkMetadata, None, None]:
        quotation_mark = quote_match.get_quotation_mark()

        if self.is_opening_quote(quote_match, previous_mark, next_mark):
            if self.is_quotation_continuer(quote_match, previous_mark, next_mark):
                quote = self.process_quotation_continuer(quote_match)
                yield quote
            else:
                if self.current_depth >= 4:
                    return

                quote = self.process_opening_mark(quote_match)
                yield quote
        elif self.is_apostrophe(quote_match, previous_mark, next_mark):
            pass
        elif self.is_closing_quote(quote_match, previous_mark, next_mark):
            if self.current_depth == 0:
                return
            quote = self.process_closing_mark(quote_match)
            yield quote
        elif self.is_malformed_closing_quote(quote_match, previous_mark, next_mark):
            quote = self.process_closing_mark(quote_match)
            yield quote
        elif self.is_malformed_opening_quote(quote_match, previous_mark, next_mark):
            quote = self.process_opening_mark(quote_match)
            yield quote

    def process_quotation_continuer(self, quote_match: QuotationMarkMatch) -> QuotationMarkMetadata:
        quote = quote_match.resolve(
            self.quotation_stack[len(self.quotation_continuer_stack)].depth, QuotationDirection.Opening
        )
        self.quotation_continuer_stack.append(quote)
        if len(self.quotation_continuer_stack) == len(self.quotation_stack):
            self.quotation_continuer_stack.clear()
        return quote

    def process_opening_mark(self, quote_match: QuotationMarkMatch) -> QuotationMarkMetadata:
        quote = quote_match.resolve(self.current_depth + 1, QuotationDirection.Opening)
        self.quotation_stack.append(quote)
        self.current_depth += 1
        return quote

    def process_closing_mark(self, quote_match: QuotationMarkMatch) -> QuotationMarkMetadata:
        quote = quote_match.resolve(self.current_depth, QuotationDirection.Closing)
        self.quotation_stack.pop()
        self.current_depth -= 1
        return quote

    def is_opening_quote(
        self,
        match: QuotationMarkMatch,
        previous_match: QuotationMarkMatch | None,
        next_match: QuotationMarkMatch | None,
    ) -> bool:

        if self.quote_convention_set.is_valid_opening_quotation_mark(match.get_quotation_mark()):
            # if the quote convention is ambiguous, use whitespace as a clue
            if self.quote_convention_set.is_valid_closing_quotation_mark(match.get_quotation_mark()):
                return (
                    self.has_leading_whitespace(match)
                    or self.has_leading_opening_quotation_mark(match)
                    or self.has_leading_colon_or_comma(match)
                ) and not (self.has_trailing_whitespace(match) or self.has_trailing_punctuation(match))
            else:
                return True
        else:
            return False

    def is_closing_quote(
        self,
        match: QuotationMarkMatch,
        previous_match: QuotationMarkMatch | None,
        next_match: QuotationMarkMatch | None,
    ) -> bool:

        if self.quote_convention_set.is_valid_closing_quotation_mark(match.get_quotation_mark()):
            # if the quote convention is ambiguous, use whitespace as a clue
            if self.quote_convention_set.is_valid_opening_quotation_mark(match.get_quotation_mark()):
                return (
                    self.has_trailing_whitespace(match)
                    or self.has_trailing_punctuation(match)
                    or self.has_trailing_closing_quotation_mark(match)
                ) and not self.has_leading_whitespace(match)
            else:
                return True
        else:
            return False

    def is_malformed_opening_quote(
        self,
        match: QuotationMarkMatch,
        previous_match: QuotationMarkMatch | None,
        next_match: QuotationMarkMatch | None,
    ) -> bool:
        if not self.quote_convention_set.is_valid_opening_quotation_mark(match.get_quotation_mark()):
            return False

        if self.has_leading_colon_or_comma(match):
            return True

        if (
            self.has_leading_whitespace(match)
            and self.has_trailing_whitespace(match)
            and len(self.quotation_stack) == 0
        ):
            return True

        return False

    def is_malformed_closing_quote(
        self,
        match: QuotationMarkMatch,
        previous_match: QuotationMarkMatch | None,
        next_match: QuotationMarkMatch | None,
    ) -> bool:
        if not self.quote_convention_set.is_valid_closing_quotation_mark(match.get_quotation_mark()):
            return False

        return (
            (
                not self.has_trailing_whitespace(match)
                or self.has_leading_whitespace(match)
                and self.has_trailing_whitespace(match)
            )
            and len(self.quotation_stack) > 0
            and self.quote_convention_set.are_marks_a_valid_pair(
                self.quotation_stack[-1].get_quotation_mark(), match.get_quotation_mark()
            )
        )

    def has_leading_whitespace(self, match: QuotationMarkMatch) -> bool:
        if match.get_previous_character() is None:
            if (
                match.get_text_segment().get_immediate_preceding_context() == UsfmMarkerType.ParagraphMarker
                or match.get_text_segment().get_immediate_preceding_context() == UsfmMarkerType.EmbedMarker
                or match.get_text_segment().get_immediate_preceding_context() == UsfmMarkerType.VerseMarker
            ):
                return True
            return False
        elif match.previous_character_matches(self.whitespace_pattern):
            return True
        return False

    def has_leading_opening_quotation_mark(self, match: QuotationMarkMatch) -> bool:
        if len(self.quotation_stack) == 0:
            return False

        return self.quotation_stack[-1].get_quotation_mark() == match.get_previous_character()

    def has_trailing_whitespace(self, match: QuotationMarkMatch) -> bool:
        return match.get_next_character() is None or match.next_character_matches(self.whitespace_pattern)

    def has_leading_colon_or_comma(self, match: QuotationMarkMatch) -> bool:
        return match.get_previous_character() is not None and (
            match.get_previous_character() == ":" or match.get_previous_character() == ","
        )

    def has_trailing_punctuation(self, match: QuotationMarkMatch) -> bool:
        return match.get_next_character() is not None and match.next_character_matches(self.punctuation_pattern)

    def has_trailing_closing_quotation_mark(self, match: QuotationMarkMatch) -> bool:
        return match.get_next_character() is not None and match.next_character_matches(self.quote_pattern)

    def is_apostrophe(
        self,
        match: QuotationMarkMatch,
        previous_match: QuotationMarkMatch | None,
        next_match: QuotationMarkMatch | None,
    ) -> bool:
        if not match.quotation_mark_matches(self.apostrophe_pattern):
            return False

        # letters on both sides of punctuation mark
        if (
            match.get_previous_character() is not None
            and match.previous_character_matches(self.latin_letter_pattern)
            and match.get_next_character() is not None
            and match.next_character_matches(self.latin_letter_pattern)
        ):
            return True

        # potential final s possessive (e.g. Moses')
        if (
            match.get_previous_character() is not None
            and match.get_previous_character() == "s"
            and (self.has_trailing_whitespace(match) or self.has_trailing_punctuation(match))
        ):
            # check whether it could be a closing quote
            if self.current_depth == 0:
                return True
            if not self.quote_convention_set.are_marks_a_valid_pair(
                self.quotation_stack[-1].get_quotation_mark(), match.get_quotation_mark()
            ):
                return True
            if next_match is not None and self.quote_convention_set.are_marks_a_valid_pair(
                self.quotation_stack[-1].get_quotation_mark(), next_match.get_quotation_mark()
            ):
                return True

        # for languages that use apostrophes at the start and end of words
        if (
            len(self.quotation_stack) == 0
            and match.get_quotation_mark() == "'"
            or len(self.quotation_stack) > 0
            and not self.quote_convention_set.are_marks_a_valid_pair(
                self.quotation_stack[-1].get_quotation_mark(), match.get_quotation_mark()
            )
        ):
            return True

        return False


class QuotationMarkTabulator:

    class QuotationMarkCounts:
        def __init__(self):
            self.string_counts = dict()
            self.total_count = 0

        def count_quotation_mark(self, quotation_mark: str) -> None:
            if quotation_mark not in self.string_counts:
                self.string_counts[quotation_mark] = 0
            self.string_counts[quotation_mark] += 1
            self.total_count += 1

        def get_best_proportion(self) -> tuple[str, int, int]:
            best_str = max(self.string_counts, key=self.string_counts.get)
            return (best_str, self.string_counts[best_str], self.total_count)

        def calculate_num_differences(self, expected_quotation_mark: str) -> int:
            if expected_quotation_mark not in self.string_counts:
                return self.total_count
            return self.total_count - self.string_counts[expected_quotation_mark]

        def get_observed_count(self) -> int:
            return self.total_count

    def __init__(self):
        self.quotation_counts_by_depth_and_direction: dict[
            tuple[int, QuotationDirection], QuotationMarkTabulator.QuotationMarkCounts
        ] = dict()

    def tabulate(self, quotation_marks: list[QuotationMarkMetadata]) -> None:
        for quotation_mark in quotation_marks:
            self.count_quotation_mark(quotation_mark)

    def count_quotation_mark(self, quote: QuotationMarkMetadata) -> None:
        key = (quote.get_depth(), quote.get_direction())
        quotation_mark = quote.get_quotation_mark()
        if key not in self.quotation_counts_by_depth_and_direction:
            self.quotation_counts_by_depth_and_direction[key] = QuotationMarkTabulator.QuotationMarkCounts()
        self.quotation_counts_by_depth_and_direction[key].count_quotation_mark(quotation_mark)

    def has_depth_and_direction_been_observed(self, depth: int, direction: QuotationDirection) -> bool:
        return (depth, direction) in self.quotation_counts_by_depth_and_direction

    def get_most_common_quote_by_depth_and_direction(
        self, depth: int, direction: QuotationDirection
    ) -> tuple[str, int, int]:
        return self.quotation_counts_by_depth_and_direction[(depth, direction)].get_best_proportion()

    def calculate_similarity(self, quote_convention: QuoteConvention) -> float:
        num_differences = 0
        num_total_quotation_marks = 0
        for depth, direction in self.quotation_counts_by_depth_and_direction:
            expected_quotation_mark: str = quote_convention.get_expected_quotation_mark(depth, direction)

            # give higher weight to shallower depths, since deeper marks are more likely to be mistakes
            num_differences += self.quotation_counts_by_depth_and_direction[
                (depth, direction)
            ].calculate_num_differences(expected_quotation_mark) * 2 ** (-depth)
            num_total_quotation_marks += self.quotation_counts_by_depth_and_direction[
                (depth, direction)
            ].get_observed_count() * 2 ** (-depth)

        if num_total_quotation_marks == 0:
            return 0
        return 1 - (num_differences / num_total_quotation_marks)

    def print_summary(self) -> None:
        for depth in range(1, 5):
            if self.has_depth_and_direction_been_observed(
                depth, QuotationDirection.Opening
            ) and self.has_depth_and_direction_been_observed(depth, QuotationDirection.Closing):
                (opening_quotation_mark, observed_opening_count, total_opening_count) = (
                    self.get_most_common_quote_by_depth_and_direction(depth, QuotationDirection.Opening)
                )
                (closing_quotation_mark, observed_closing_count, total_closing_count) = (
                    self.get_most_common_quote_by_depth_and_direction(depth, QuotationDirection.Closing)
                )
                print(
                    "The most common level %i quotes are %s (%i of %i opening quotes) and %s (%i of %i closing quotes)"
                    % (
                        depth,
                        opening_quotation_mark,
                        observed_opening_count,
                        total_opening_count,
                        closing_quotation_mark,
                        observed_closing_count,
                        total_closing_count,
                    )
                )


class UsfmVerseTextExtractor(UsfmParserHandler):
    def __init__(self):
        self.text_segments: list[TextSegment] = []
        self.next_text_segment_builder: TextSegment.Builder = TextSegment.Builder()

    def chapter(
        self,
        state: UsfmParserState,
        number: str,
        marker: str,
        alt_number: Optional[str],
        pub_number: Optional[str],
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.ChapterMarker)

    def start_para(
        self,
        state: UsfmParserState,
        marker: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.ParagraphMarker)

    def start_char(
        self,
        state: UsfmParserState,
        marker_without_plus: str,
        unknown: bool,
        attributes: Optional[Sequence[UsfmAttribute]],
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.CharacterMarker)

    def end_char(
        self, state: UsfmParserState, marker: str, attributes: Optional[Sequence[UsfmAttribute]], closed: bool
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.CharacterMarker)

    def verse(
        self, state: UsfmParserState, number: str, marker: str, alt_number: Optional[str], pub_number: Optional[str]
    ) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.VerseMarker)

    def end_note(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EmbedMarker)

    def end_table(self, state: UsfmParserState) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EmbedMarker)

    def ref(self, state: UsfmParserState, marker: str, display: str, target: str) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EmbedMarker)

    def end_sidebar(self, state: UsfmParserState, marker: str, closed: bool) -> None:
        self.next_text_segment_builder.add_preceding_marker(UsfmMarkerType.EmbedMarker)

    def text(self, state: UsfmParserState, text: str) -> None:
        if not state.is_verse_text:
            return
        if len(text) > 0:
            self.next_text_segment_builder.set_text(text)
            text_segment: TextSegment = self.next_text_segment_builder.build()
            if len(self.text_segments) > 0:
                self.text_segments[-1].set_next_segment(text_segment)
            self.text_segments.append(text_segment)
        self.next_text_segment_builder = TextSegment.Builder()

    def get_chapters(self) -> list[Chapter]:
        chapters: list[Chapter] = []
        current_chapter_verses: list[Verse] = []
        current_verse_segments: list[TextSegment] = []
        for text_segment in self.text_segments:
            if text_segment.is_marker_in_preceding_context(UsfmMarkerType.VerseMarker):
                if len(current_verse_segments) > 0:
                    current_chapter_verses.append(Verse(current_verse_segments))
                current_verse_segments = []
            if text_segment.is_marker_in_preceding_context(UsfmMarkerType.ChapterMarker):
                if len(current_chapter_verses) > 0:
                    chapters.append(Chapter(current_chapter_verses))
                current_chapter_verses = []
            current_verse_segments.append(text_segment)
        if len(current_verse_segments) > 0:
            current_chapter_verses.append(Verse(current_verse_segments))
        if len(current_chapter_verses) > 0:
            chapters.append(Chapter(current_chapter_verses))
        return chapters


standard_quote_conventions: QuoteConventionSet = QuoteConventionSet(
    [
        QuoteConvention(
            "standard_english",
            [
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
            ],
        ),
        QuoteConvention(
            "typewriter_english",
            [
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
            ],
        ),
        QuoteConvention(
            "british_english",
            [
                SingleLevelQuoteConvention("\u2018", "\u2019"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
            ],
        ),
        QuoteConvention(
            "british_typewriter_english",
            [
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
            ],
        ),
        QuoteConvention(
            "hybrid_typewriter_english",
            [
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
            ],
        ),
        QuoteConvention(
            "standard_french",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u2039", "\u203a"),
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u2039", "\u203a"),
            ],
        ),
        QuoteConvention(
            "typewriter_french",
            [
                SingleLevelQuoteConvention("<<", ">>"),
                SingleLevelQuoteConvention("<", ">"),
                SingleLevelQuoteConvention("<<", ">>"),
                SingleLevelQuoteConvention("<", ">"),
            ],
        ),
        QuoteConvention(
            "french_variant",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u2039", "\u203a"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
            ],
        ),
        QuoteConvention(
            "western_european",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
            ],
        ),
        QuoteConvention(
            "british_inspired_western_european",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u2018", "\u2019"),
                SingleLevelQuoteConvention("\u201c", "\u201d"),
            ],
        ),
        QuoteConvention(
            "typewriter_western_european",
            [
                SingleLevelQuoteConvention("<<", ">>"),
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
            ],
        ),
        QuoteConvention(
            "typewriter_western_european_variant",
            [
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("<", ">"),
                SingleLevelQuoteConvention("'", "'"),
            ],
        ),
        QuoteConvention(
            "hybrid_typewriter_western_european",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention('"', '"'),
                SingleLevelQuoteConvention("'", "'"),
            ],
        ),
        QuoteConvention(
            "hybrid_british_typewriter_western_european",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("'", "'"),
                SingleLevelQuoteConvention('"', '"'),
            ],
        ),
        QuoteConvention(
            "standard_german",
            [
                SingleLevelQuoteConvention("\u00bb", "\u00ab"),
                SingleLevelQuoteConvention("\u203a", "\u2039"),
                SingleLevelQuoteConvention("\u00bb", "\u00ab"),
                SingleLevelQuoteConvention("\u203a", "\u2039"),
            ],
        ),
        QuoteConvention(
            "newspaper_german",
            [
                SingleLevelQuoteConvention("\u201e", "\u201c"),
                SingleLevelQuoteConvention("\u201a", "\u2018"),
                SingleLevelQuoteConvention("\u201e", "\u201c"),
                SingleLevelQuoteConvention("\u201a", "\u2018"),
            ],
        ),
        QuoteConvention(
            "standard_swedish",
            [
                SingleLevelQuoteConvention("\u201d", "\u201d"),
                SingleLevelQuoteConvention("\u2019", "\u2019"),
                SingleLevelQuoteConvention("\u201d", "\u201d"),
                SingleLevelQuoteConvention("\u2019", "\u2019"),
            ],
        ),
        QuoteConvention(
            "standard_finnish",
            [
                SingleLevelQuoteConvention("\u00bb", "\u00bb"),
                SingleLevelQuoteConvention("\u2019", "\u2019"),
            ],
        ),
        QuoteConvention(
            "standard_hungarian",
            [
                SingleLevelQuoteConvention("\u201e", "\u201d"),
                SingleLevelQuoteConvention("\u201a", "\u2019"),
                SingleLevelQuoteConvention("\u201e", "\u201d"),
                SingleLevelQuoteConvention("\u201a", "\u2019"),
            ],
        ),
        QuoteConvention(
            "eastern_european",
            [
                SingleLevelQuoteConvention("\u201e", "\u201c"),
                SingleLevelQuoteConvention("\u201a", "\u2018"),
                SingleLevelQuoteConvention("\u201e", "\u201c"),
                SingleLevelQuoteConvention("\u201a", "\u2018"),
            ],
        ),
        QuoteConvention(
            "standard_russian",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u201e", "\u201c"),
                SingleLevelQuoteConvention("\u201a", "\u2018"),
            ],
        ),
        QuoteConvention(
            "standard_arabic",
            [
                SingleLevelQuoteConvention("\u201d", "\u201c"),
                SingleLevelQuoteConvention("\u2019", "\u2018"),
                SingleLevelQuoteConvention("\u201d", "\u201c"),
                SingleLevelQuoteConvention("\u2019", "\u2018"),
            ],
        ),
        QuoteConvention(
            "non-standard_arabic",
            [
                SingleLevelQuoteConvention("\u00ab", "\u00bb"),
                SingleLevelQuoteConvention("\u2019", "\u2018"),
            ],
        ),
    ]
)


class QuoteConventionAnalysis:
    def __init__(self, best_quote_convention: QuoteConvention, best_quote_convention_score: float):
        self.best_quote_convention = best_quote_convention
        self.best_quote_convention_score = best_quote_convention_score

    def get_best_quote_convention(self) -> QuoteConvention:
        return self.best_quote_convention

    def get_best_quote_convention_similarity_score(self) -> float:
        return self.best_quote_convention_score * 100


def _analyze_quote_convention_for_chapters(
    chapters: list[Chapter], print_summary: bool = True
) -> Union[QuoteConventionAnalysis, None]:
    updated_quotation_config = PreliminaryQuotationAnalyzer(
        standard_quote_conventions
    ).narrow_down_possible_quote_conventions(chapters)

    quotation_mark_tabulator = QuotationMarkTabulator()
    for chapter in chapters:
        quotation_mark_matches: list[QuotationMarkMatch] = QuotationMarkFinder(
            updated_quotation_config
        ).find_all_potential_quotation_marks_in_chapter(chapter)
        resolved_quotation_marks: list[QuotationMarkMetadata] = QuotationMarkResolver(
            updated_quotation_config
        ).resolve_quotation_marks(quotation_mark_matches)
        quotation_mark_tabulator.tabulate(resolved_quotation_marks)

    (best_quote_convention, score) = standard_quote_conventions.find_most_similar_convention(quotation_mark_tabulator)

    if print_summary:
        quotation_mark_tabulator.print_summary()

    if score > 0:
        return QuoteConventionAnalysis(best_quote_convention, score)
    return None


def analyze_project_quote_convention(
    paratext_project_name: str, corpus_books: Dict[int, List[int]]
) -> Union[QuoteConventionAnalysis, None]:
    verse_text_extractor = UsfmVerseTextExtractor()
    parse_books(SIL_NLP_ENV.pt_projects_dir / paratext_project_name, corpus_books, verse_text_extractor)
    return _analyze_quote_convention_for_chapters(verse_text_extractor.get_chapters())


def analyze_experiment_target_quote_convention(
    experiment_name: str, corpus_books: Union[Dict[int, List[int]], None] = None
) -> Union[QuoteConventionAnalysis, None]:
    config: Config = load_config(experiment_name)
    for pair in config.corpus_pairs:
        if not pair.is_scripture or pair.type == DataFileType.TEST or pair.type == DataFileType.VAL:
            continue
        for target_file in pair.trg_files:
            paratext_project_name = target_file.project

            return analyze_project_quote_convention(paratext_project_name, corpus_books or pair.corpus_books)


def analyze_usfm_quote_convention(
    usfm_path: Path, encoding: Union[str, None] = None, print_summary: bool = False
) -> Union[QuoteConventionAnalysis, None]:
    with open(usfm_path, "r", encoding=encoding or "utf-8") as usfm_input:
        verse_text_extractor = UsfmVerseTextExtractor()
        parse_usfm(usfm_input.read(), verse_text_extractor)
        return _analyze_quote_convention_for_chapters(verse_text_extractor.get_chapters(), print_summary)


def main() -> None:

    # Allow options for 1) experiment folder, 2) USFM file, 3) Paratext project + book list
    parser = argparse.ArgumentParser(
        prog="analyze_quote_convention",
        description="Determines the quote convention that is used in a Paratext project",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Name of an experiment folder. The experiment's target project will be analyzed.",
    )
    parser.add_argument("--project", type=str, default=None, help="Name of a Paratext folder.")
    parser.add_argument("--usfm-file", type=Path, default=None, help="Path of an individual USFM file")
    parser.add_argument("--encoding", type=str, default=None, help="Encoding of the file specified with --usfm-file")
    parser.add_argument(
        "--books",
        type=str,
        default=None,
        help="List of books to analyze (with --experiment or --project). The format should match corpus_books in config.yml.",
    )
    args = parser.parse_args()

    quote_convention_analysis: Union[QuoteConventionAnalysis, None] = None
    if args.experiment is not None:
        if args.project is not None:
            LOGGER.warning("Ignoring --project since --experiment was also specified.")
        if args.usfm_file is not None:
            LOGGER.warning("Ignoring --usfm-file since --experiment was also specified.")
        if args.books is not None:
            selected_books = get_chapters(args.books)
            quote_convention_analysis = analyze_experiment_target_quote_convention(args.experiment, selected_books)
        else:
            quote_convention_analysis = analyze_experiment_target_quote_convention(args.experiment)

    elif args.project is not None:
        if args.usfm_file is not None:
            LOGGER.warning("Ignoring --usfm-file since --project was also specified.")
        if args.books is None:
            LOGGER.error("When using --project, you must specify --books.")
            return
        selected_books = get_chapters(args.books)
        quote_convention_analysis = analyze_project_quote_convention(args.project, selected_books)

    elif args.usfm_file is not None:
        quote_convention_analysis = analyze_usfm_quote_convention(args.usfm_file, args.encoding)

    else:
        LOGGER.error("One of --experiment, --project, or --usfm-file must be specified")
        return

    if quote_convention_analysis is not None:
        print("======================")
        print("Best quote convention:")
        print("----------------------")
        quote_convention_analysis.get_best_quote_convention().print_summary()
        print("======================")
        print("Similarity = %.2f%%" % quote_convention_analysis.get_best_quote_convention_similarity_score())
    else:
        print("No quote convention was detected")


if __name__ == "__main__":
    main()
