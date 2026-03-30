from abc import ABC, abstractmethod
from typing import List, Tuple

from .utils import contains_letter
from .word_alignments import WordAlignments


class BreakScorer(ABC):
    def __init__(self, source_tokens: List[str], target_tokens: List[str], word_alignments: WordAlignments):
        self._source_tokens = source_tokens
        self._target_tokens = target_tokens
        self._word_alignments = word_alignments

    @abstractmethod
    def score_potential_break(
        self, src_break_offset: int, trg_break_offset: int, src_range: Tuple[int, int] | None = None
    ) -> float: ...


class FewestCrossedAlignmentsBreakScorer(BreakScorer):
    def score_potential_break(
        self, src_break_offset: int, trg_break_offset: int, src_range: Tuple[int, int] | None = None
    ) -> float:
        num_crossed_alignments = self._word_alignments.get_num_crossed_alignments(src_break_offset, trg_break_offset)
        return -num_crossed_alignments


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
