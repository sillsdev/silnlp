from abc import ABC, abstractmethod
from typing import List, MutableSet, Tuple

from silnlp.alignment.verse_segmentation.break_scorers import (
    AbstractBreakScorerFactory,
    BreakScorer,
    FewestCrossedAlignmentsBreakScorer,
)
from silnlp.alignment.verse_segmentation.utils import contains_letter
from silnlp.alignment.verse_segmentation.word_alignments import WordAlignments


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


class AbstractVerseOffsetPredictorFactory(ABC):
    @abstractmethod
    def create(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
    ) -> VerseOffsetPredictor: ...


class ScoringFunctionVerseOffsetPredictor(VerseOffsetPredictor):
    def __init__(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
        source_verse_token_offsets: List[int],
        word_alignments: WordAlignments,
        break_scorer: BreakScorer,
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
        self._target_verse_token_offsets: List[int] = [-1 for _ in self._source_verse_token_offsets]

        if self._source_verse_token_offsets[-1] == num_src_tokens:
            self._target_verse_token_offsets[-1] = num_trg_tokens

        self._break_scorer = break_scorer

    def _add_trg_verse_break(self, src_offset: int, trg_offset: int) -> None:
        self._remaining_breaks_to_map.remove(src_offset)
        for index, source_verse_token_offset in enumerate(self._source_verse_token_offsets):
            if source_verse_token_offset == src_offset:
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


class FewestCrossedAlignmentsVerseOffsetPredictorFactory(AbstractVerseOffsetPredictorFactory):
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
            FewestCrossedAlignmentsBreakScorer(source_tokens, target_tokens, word_alignments),
        )


class AdaptedMarkerPlacementVerseOffsetPredictor(VerseOffsetPredictor):

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
