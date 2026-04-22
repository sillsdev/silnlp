from dataclasses import dataclass
from typing import Any, Callable, List, cast

import torch
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

from silnlp.nmt.hugging_face_config import HuggingFaceConfig, PreTrainedModelProvider, PreTrainedModelProviderFactory


@dataclass
class MockModelOutput:
    sequences: list[torch.Tensor]
    scores: list[torch.Tensor]
    sequences_scores: list[torch.Tensor]


class MockPretrainedModel(PreTrainedModel):
    supports_gradient_checkpointing = True
    _num_forward_calls: int = 0
    _observed_training_batch_sizes: List[int] = []
    _total_number_of_training_data_elements: int = 0

    def __init__(self, outputs: List[GenerateBeamEncoderDecoderOutput], transition_scores: List[torch.Tensor]) -> None:
        super().__init__(PretrainedConfig(decoder_start_token_id=1))
        self._outputs = outputs
        self._transition_scores = transition_scores
        self._generate_call_count = 0
        self._transition_score_call_count = 0
        self._input_embeddings = torch.nn.Embedding(1, 1)
        self._position_embeddings = torch.nn.Embedding(1, 1)
        self.generation_config = GenerationConfig(min_length=0, max_length=512, pad_token_id=1)

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def get_position_embeddings(self) -> torch.nn.Embedding | tuple[torch.nn.Embedding]:
        return self._position_embeddings

    def get_input_embeddings(self) -> torch.nn.Module:
        return self._input_embeddings

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> torch.nn.Embedding:
        _ = (new_num_tokens, pad_to_multiple_of, mean_resizing)
        self._input_embeddings = torch.nn.Embedding(1, 1)
        return self._input_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        _ = (attention_mask, labels, decoder_input_ids, decoder_attention_mask, kwargs)
        MockPretrainedModel._num_forward_calls += 1
        MockPretrainedModel._observed_training_batch_sizes.append(input_ids.shape[0] if input_ids is not None else 0)
        MockPretrainedModel._total_number_of_training_data_elements += (
            input_ids.shape[0] * input_ids.shape[1] if input_ids is not None else 0
        )

        # The Trainer requires a differentiable tensor for the loss
        loss = cast(torch.FloatTensor, self._input_embeddings.weight.sum().float() * 0.0)

        # Provide logits with expected rank [batch, seq, vocab]
        batch_size = 1 if input_ids is None else input_ids.size(0)
        seq_len = 1 if input_ids is None else input_ids.size(1)
        logits = cast(
            torch.FloatTensor,
            torch.zeros(batch_size, seq_len, 1, dtype=torch.float32, device=self._input_embeddings.weight.device),
        )
        return Seq2SeqLMOutput(loss=loss, logits=logits)

    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: Any | None = None,
        stopping_criteria: Any | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = None,
        synced_gpus: bool | None = None,
        assistant_model: PreTrainedModel | None = None,
        streamer: Any | None = None,
        negative_prompt_ids: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> GenerateBeamEncoderDecoderOutput:
        if self._generate_call_count >= len(self._outputs):
            raise AssertionError("Mock model generate was called more times than expected.")
        output = self._outputs[self._generate_call_count]
        self._generate_call_count += 1
        return output

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores,
        beam_indices=None,
        normalize_logits: bool = True,
    ) -> torch.Tensor:
        _ = (sequences, beam_indices, normalize_logits)
        if self._transition_score_call_count >= len(self._transition_scores):
            raise AssertionError("Mock model compute_transition_scores was called more times than expected.")
        scores = self._transition_scores[self._transition_score_call_count]
        self._transition_score_call_count += 1
        return scores

    @classmethod
    def get_num_forward_calls(cls) -> int:
        return cls._num_forward_calls

    @classmethod
    def get_observed_training_batch_sizes(cls) -> list[int]:
        return cls._observed_training_batch_sizes

    @classmethod
    def get_total_number_of_training_data_elements(cls) -> int:
        return cls._total_number_of_training_data_elements


def _create_mock_pretrained_model(mock_output: MockModelOutput) -> MockPretrainedModel:
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
    transition_scores = [scores.squeeze(1) if scores.ndim == 3 else scores for scores in mock_output.scores]
    return MockPretrainedModel(outputs, transition_scores)


class MockPretrainedModelProvider(PreTrainedModelProvider):
    def __init__(self, mock_outputs: List[MockModelOutput]):
        self._mock_outputs = mock_outputs
        self._current_output_index = 0

    def create_model_for_training(
        self, model_name: str, model_config: PretrainedConfig, device_map: dict[str, int]
    ) -> PreTrainedModel:
        return MockPretrainedModel([], [])

    def create_model_for_inference(self, model_name: str) -> PreTrainedModel:
        if self._current_output_index >= len(self._mock_outputs):
            raise AssertionError("Mock model provider was called more times than expected.")
        mock_output = self._mock_outputs[self._current_output_index]
        self._current_output_index += 1
        return _create_mock_pretrained_model(mock_output)


class MockPreTrainedModelProviderFactory(PreTrainedModelProviderFactory):
    def __init__(self, mock_outputs: List[MockModelOutput]):
        self._mock_outputs = mock_outputs

    def create_pretrained_model_provider(
        self, config: HuggingFaceConfig, mixed_precision: bool = False
    ) -> PreTrainedModelProvider:
        return MockPretrainedModelProvider(self._mock_outputs)
