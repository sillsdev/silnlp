from dataclasses import dataclass, field
from typing import List, Optional, cast

import torch
from transformers import LlamaForCausalLM, PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput

from silnlp.nmt.experiment_settings import TrainerSettings
from silnlp.nmt.finetune_method import FinetuneMethod
from silnlp.nmt.llm_config import CausalLMProvider, CausalLMProviderFactory

_TINY_MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# A short, fixed block of generated token ids appended to every prompt during mock inference.
# The exact ids are unimportant - the tests assert structural properties (one output line per
# input sentence), not the decoded text, which depends on the random tiny model's tokenizer.
_MOCK_NEW_TOKEN_IDS = [10, 11, 12, 13]


@dataclass
class CausalModelTrainingStats:
    num_forward_calls: int = 0
    observed_training_batch_sizes: List[int] = field(default_factory=list)

    def record_forward(self, input_ids: Optional[torch.Tensor]) -> None:
        self.num_forward_calls += 1
        if input_ids is not None:
            self.observed_training_batch_sizes.append(input_ids.shape[0])


class StandInCausalModel(LlamaForCausalLM):
    """The tiny test model, recording the batches it is trained on."""

    @classmethod
    def create(cls, stats: CausalModelTrainingStats) -> "StandInCausalModel":
        model = cast("StandInCausalModel", cls.from_pretrained(_TINY_MODEL_NAME))
        model.stats = stats
        return model

    # The Trainer reads this signature twice: to choose which dataset columns to keep, and to find
    # the label argument, so labels has to be named here or the labels column is dropped.
    def forward(self, input_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, *args, **kwargs):
        self.stats.record_forward(input_ids)
        return super().forward(input_ids=input_ids, labels=labels, *args, **kwargs)


class GeneratingStandInCausalModel(StandInCausalModel):
    """The stand-in used for inference, which appends a fixed block of tokens to every prompt."""

    def generate(self, *args, **kwargs) -> GenerateDecoderOnlyOutput:
        input_ids: torch.Tensor = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        appended = torch.tensor([_MOCK_NEW_TOKEN_IDS], dtype=torch.long, device=input_ids.device).repeat(batch_size, 1)
        sequences = torch.cat([input_ids, appended], dim=1)
        return GenerateDecoderOnlyOutput(sequences=cast(torch.LongTensor, sequences))


class MockCausalLMProvider(CausalLMProvider):
    def __init__(
        self,
        model: str,
        params: dict,
        finetuning: FinetuneMethod,
        trainer_settings: TrainerSettings,
        mixed_precision: bool,
        stats: CausalModelTrainingStats,
    ):
        super().__init__(model, params, finetuning, trainer_settings, mixed_precision)
        self._stats = stats

    def create_model_for_training(self) -> PreTrainedModel:
        return StandInCausalModel.create(self._stats)

    def create_model_for_inference(self, checkpoint_path) -> PreTrainedModel:
        return GeneratingStandInCausalModel.create(self._stats)


class MockCausalLMProviderFactory(CausalLMProviderFactory):
    def __init__(self, stats: Optional[CausalModelTrainingStats] = None):
        self._stats = stats or CausalModelTrainingStats()

    @property
    def stats(self) -> CausalModelTrainingStats:
        return self._stats

    def create(
        self,
        model: str,
        params: dict,
        finetuning: FinetuneMethod,
        trainer_settings: TrainerSettings,
        mixed_precision: bool,
    ) -> CausalLMProvider:
        return MockCausalLMProvider(model, params, finetuning, trainer_settings, mixed_precision, self._stats)
