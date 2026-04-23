from typing import List

from .break_scorers import AbstractBreakScorerFactory, BreakScorer
from .sub_passage import SubPassage
from .word_alignments import WordAlignments


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
