from dataclasses import dataclass, field
from typing import List, Optional, cast
from unittest.mock import Mock, create_autospec

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput

from silnlp.nmt.llm_config import CausalLMProvider, CausalLMProviderFactory, LLMConfig

_TINY_MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# A short, fixed block of generated token ids appended to every prompt during mock inference.
# The exact ids are unimportant - the tests assert structural properties (one output line per
# input sentence), not the decoded text, which depends on the random tiny model's tokenizer.
_MOCK_NEW_TOKEN_IDS = [10, 11, 12, 13]


@dataclass
class CausalModelTrainingStats:
    num_forward_calls: int = 0
    observed_training_batch_sizes: List[int] = field(default_factory=list)


def _build_training_model(stats: CausalModelTrainingStats) -> PreTrainedModel:
    model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(_TINY_MODEL_NAME))
    underlying_forward = model.forward

    def mock_forward(input_ids: Optional[torch.Tensor] = None, *args, **kwargs):
        stats.num_forward_calls += 1
        if input_ids is not None:
            stats.observed_training_batch_sizes.append(input_ids.shape[0])
        return underlying_forward(input_ids=input_ids, *args, **kwargs)

    model.forward = create_autospec(mock_forward, side_effect=mock_forward)
    return model


def _build_inference_model(stats: CausalModelTrainingStats) -> PreTrainedModel:
    model = _build_training_model(stats)

    def mock_generate(*args, **kwargs) -> GenerateDecoderOnlyOutput:
        input_ids: torch.Tensor = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        appended = torch.tensor([_MOCK_NEW_TOKEN_IDS], dtype=torch.long, device=input_ids.device).repeat(batch_size, 1)
        sequences = torch.cat([input_ids, appended], dim=1)
        return GenerateDecoderOnlyOutput(sequences=cast(torch.LongTensor, sequences))

    model.generate = Mock(side_effect=mock_generate)
    return model


class MockCausalLMProvider(CausalLMProvider):
    def __init__(self, config: LLMConfig, mixed_precision: bool, stats: CausalModelTrainingStats):
        super().__init__(config, mixed_precision)
        self._stats = stats

    def create_model_for_training(self) -> PreTrainedModel:
        return _build_training_model(self._stats)

    def create_model_for_inference(self, checkpoint_path) -> PreTrainedModel:
        return _build_inference_model(self._stats)


class MockCausalLMProviderFactory(CausalLMProviderFactory):
    def __init__(self, stats: Optional[CausalModelTrainingStats] = None):
        self._stats = stats or CausalModelTrainingStats()

    @property
    def stats(self) -> CausalModelTrainingStats:
        return self._stats

    def create(self, config: LLMConfig, mixed_precision: bool) -> CausalLMProvider:
        return MockCausalLMProvider(config, mixed_precision, self._stats)
