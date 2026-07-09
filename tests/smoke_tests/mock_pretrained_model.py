from dataclasses import dataclass, field
from typing import Iterator, List, cast
from unittest.mock import Mock, create_autospec

import torch
from transformers import AutoModelForSeq2SeqLM, PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

from silnlp.nmt.seq2seq_config import PreTrainedModelProvider, PreTrainedModelProviderFactory, Seq2SeqConfig

_TINY_MODEL_NAME = "hf-internal-testing/tiny-random-nllb"


@dataclass
class MockModelOutput:
    sequences: list[torch.Tensor]
    scores: list[torch.Tensor]
    sequences_scores: list[torch.Tensor]


@dataclass
class ModelTrainingStats:
    num_forward_calls: int = 0
    observed_training_batch_sizes: list[int] = field(default_factory=list)
    total_number_of_training_data_elements: int = 0


def create_mock_pretrained_model(
    output_iterator: Iterator[GenerateBeamEncoderDecoderOutput], model_stats: ModelTrainingStats | None = None
) -> PreTrainedModel:
    if model_stats is None:
        model_stats = ModelTrainingStats()

    underlying_model = cast(PreTrainedModel, AutoModelForSeq2SeqLM.from_pretrained(_TINY_MODEL_NAME))
    underlying_model_forward = underlying_model.forward
    last_transition_scores: torch.Tensor | None = None

    def mock_forward(
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        model_stats.num_forward_calls += 1
        model_stats.observed_training_batch_sizes.append(input_ids.shape[0] if input_ids is not None else 0)
        model_stats.total_number_of_training_data_elements += (
            input_ids.shape[0] * input_ids.shape[1] if input_ids is not None else 0
        )
        kwargs.pop("num_items_in_batch", None)
        return underlying_model_forward(
            input_ids=input_ids,
            *args,
            **kwargs,
        )

    def mock_generate(*args, **kwargs) -> GenerateBeamEncoderDecoderOutput:
        nonlocal last_transition_scores
        output = next(output_iterator)

        # SilTranslationPipeline calls compute_transition_scores and requires a shape that
        # matches the generated sequences/scores
        assert output.scores is not None
        last_transition_scores = output.scores[0]
        return output

    def mock_compute_transition_scores(
        sequences: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert last_transition_scores is not None
        return last_transition_scores.to(sequences.device)

    underlying_model.forward = create_autospec(mock_forward, side_effect=mock_forward)
    underlying_model.generate = Mock(side_effect=mock_generate)
    underlying_model.compute_transition_scores = Mock(side_effect=mock_compute_transition_scores)
    return underlying_model


class MockPretrainedModelProvider(PreTrainedModelProvider):
    def __init__(self, mock_outputs: Iterator[MockModelOutput], model_stats: ModelTrainingStats):
        self._mock_outputs = mock_outputs
        self._model_stats = model_stats
        self._current_output_index = 0

    def create_model_for_training(
        self, model_name: str, model_config: PretrainedConfig, device_map: dict[str, int]
    ) -> PreTrainedModel:
        return create_mock_pretrained_model(iter(()), self._model_stats)

    def create_model_for_inference(self, model_name: str) -> PreTrainedModel:
        mock_output = self._prepare_outputs(next(self._mock_outputs))
        return create_mock_pretrained_model(mock_output, self._model_stats)

    def _prepare_outputs(self, mock_output: MockModelOutput) -> Iterator[GenerateBeamEncoderDecoderOutput]:
        assert len(mock_output.sequences) == len(mock_output.scores)
        assert len(mock_output.scores) == len(mock_output.sequences_scores)

        outputs = [
            GenerateBeamEncoderDecoderOutput(
                sequences=cast(torch.LongTensor, sequences),
                beam_indices=cast(torch.LongTensor, torch.zeros_like(sequences)),
                scores=(cast(torch.FloatTensor, scores),),
                sequences_scores=cast(torch.FloatTensor, sequences_scores),
            )
            for sequences, scores, sequences_scores in zip(
                mock_output.sequences, mock_output.scores, mock_output.sequences_scores
            )
        ]
        return iter(outputs)


class MockPreTrainedModelProviderFactory(PreTrainedModelProviderFactory):
    def __init__(self, mock_outputs: List[MockModelOutput], model_stats: ModelTrainingStats | None = None):
        self._mock_outputs = mock_outputs
        self._model_stats = model_stats or ModelTrainingStats()

    @property
    def stats(self) -> ModelTrainingStats:
        return self._model_stats

    def create_pretrained_model_provider(
        self, config: Seq2SeqConfig, mixed_precision: bool = False
    ) -> PreTrainedModelProvider:
        return MockPretrainedModelProvider(iter(self._mock_outputs), self._model_stats)
