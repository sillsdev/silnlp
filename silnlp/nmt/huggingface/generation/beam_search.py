from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from machine.translation import WordGraph
from torch import FloatTensor, LongTensor, Tensor
from transformers import BeamScorer
from transformers.generation import BeamSearchEncoderDecoderOutput


@dataclass
class WordGraphBeamSearchEncoderDecoderOutput(BeamSearchEncoderDecoderOutput):
    word_graphs: Optional[List[WordGraph]] = None


class WordGraphBeamScorerDecorator(BeamScorer):
    def __init__(
        self,
        scorer: BeamScorer,
        input_ids: LongTensor,
    ):
        self._scorer = scorer
        self.word_graphs = [WordGraph()]

    def process(
        self,
        input_ids: LongTensor,
        next_scores: FloatTensor,
        next_tokens: LongTensor,
        next_indices: LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[LongTensor] = None,
        group_index: Optional[int] = 0,
    ) -> Dict[str, Tensor]:
        # TODO: add arcs to word graph
        return self._scorer.process(
            input_ids, next_scores, next_tokens, next_indices, pad_token_id, eos_token_id, beam_indices, group_index
        )

    def finalize(
        self,
        input_ids: LongTensor,
        next_scores: FloatTensor,
        next_tokens: LongTensor,
        next_indices: LongTensor,
        max_length: int,
        **kwargs,
    ) -> LongTensor:
        return self._scorer.finalize(input_ids, next_scores, next_tokens, next_indices, max_length, **kwargs)
