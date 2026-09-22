from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Union, cast

import torch
from transformers import M2M100ForConditionalGeneration, PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

from silnlp.nmt.experiment_languages import ExperimentLanguages
from silnlp.nmt.model_name import ModelName
from silnlp.nmt.pretrained_tokenizer import PretrainedTokenizer
from silnlp.nmt.seq2seq_config import PreTrainedModelProvider, PreTrainedModelProviderFactory
from silnlp.nmt.translation_settings import ModelSettings

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

    def record_training_batch(self, input_ids: Optional[torch.Tensor]) -> None:
        self.num_forward_calls += 1
        self.observed_training_batch_sizes.append(input_ids.shape[0] if input_ids is not None else 0)
        self.total_number_of_training_data_elements += (
            input_ids.shape[0] * input_ids.shape[1] if input_ids is not None else 0
        )


class ScriptedGeneration(ABC):
    """What a stand-in model generates, decided by the test rather than by its weights."""

    @abstractmethod
    def outputs_for(self, input_ids: Optional[torch.Tensor]) -> GenerateBeamEncoderDecoderOutput:
        ...


class NoGeneration(ScriptedGeneration):
    def outputs_for(self, input_ids: Optional[torch.Tensor]) -> GenerateBeamEncoderDecoderOutput:
        raise AssertionError("The model was asked to generate translations, but no mock output was provided.")


class RecordedGeneration(ScriptedGeneration):
    def __init__(self, outputs: Iterator[GenerateBeamEncoderDecoderOutput]) -> None:
        self._outputs = outputs

    def outputs_for(self, input_ids: Optional[torch.Tensor]) -> GenerateBeamEncoderDecoderOutput:
        return next(self._outputs)


class FixedTranslationGeneration(ScriptedGeneration):
    """Generates the same translation for every sentence in every batch."""

    def __init__(self, translation_token_ids: "TranslationTokenIds") -> None:
        self._translation_token_ids = translation_token_ids
        self._num_translated_sentences = 0

    def outputs_for(self, input_ids: Optional[torch.Tensor]) -> GenerateBeamEncoderDecoderOutput:
        assert input_ids is not None
        batch_size = input_ids.shape[0]
        device = input_ids.device

        sequences = torch.tensor(self._translation_token_ids.resolve(), dtype=torch.long, device=device).repeat(
            batch_size, 1
        )
        scores = torch.full(tuple(sequences.shape), MOCK_TOKEN_LOG_PROB, dtype=torch.float32, device=device)
        sequences_scores = torch.tensor(
            [mock_sequence_log_prob(self._num_translated_sentences + i) for i in range(batch_size)],
            dtype=torch.float32,
            device=device,
        )
        self._num_translated_sentences += batch_size
        return GenerateBeamEncoderDecoderOutput(
            sequences=cast(torch.LongTensor, sequences),
            beam_indices=cast(torch.LongTensor, torch.zeros_like(sequences)),
            scores=(cast(torch.FloatTensor, scores),),
            sequences_scores=cast(torch.FloatTensor, sequences_scores),
        )


class ModelScript:
    """The behaviour a stand-in model is given: what it generates and what it records."""

    def __init__(self, generation: ScriptedGeneration, stats: ModelTrainingStats) -> None:
        self._generation = generation
        self._stats = stats
        self._last_transition_scores: List[torch.Tensor] = []

    def record_training_batch(self, input_ids: Optional[torch.Tensor]) -> None:
        self._stats.record_training_batch(input_ids)

    def generate(
        self, input_ids: Optional[torch.Tensor], return_dict: bool
    ) -> Union[GenerateBeamEncoderDecoderOutput, torch.Tensor]:
        output = self._generation.outputs_for(input_ids)
        # The evaluation pass during training asks for token ids alone, not the scored output.
        if not return_dict:
            return output.sequences
        assert output.scores is not None
        self._last_transition_scores = [output.scores[0]]
        return output

    def transition_scores_for(self, sequences: torch.Tensor) -> torch.Tensor:
        assert len(self._last_transition_scores) > 0
        return self._last_transition_scores[0].to(sequences.device)


class StandInSeq2SeqModel(M2M100ForConditionalGeneration):
    """The tiny test model, with its generation replaced by whatever the test scripted."""

    @classmethod
    def create(cls, generation: ScriptedGeneration, stats: ModelTrainingStats) -> "StandInSeq2SeqModel":
        model = cast("StandInSeq2SeqModel", cls.from_pretrained(_TINY_MODEL_NAME, token=False))
        model.script = ModelScript(generation, stats)
        return model

    # The Trainer reads this signature twice: to choose which dataset columns to keep, and to find
    # the label argument, so labels has to be named here or the labels column is dropped.
    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Seq2SeqLMOutput:
        # The evaluation pass runs forward too, so only the training steps are counted.
        if self.training:
            self.script.record_training_batch(input_ids)
        kwargs.pop("num_items_in_batch", None)
        return super().forward(input_ids=input_ids, labels=labels, *args, **kwargs)

    def generate(self, *args, **kwargs) -> Union[GenerateBeamEncoderDecoderOutput, torch.Tensor]:
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        return self.script.generate(input_ids, return_dict=bool(kwargs.get("return_dict_in_generate", False)))

    def compute_transition_scores(self, sequences: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.script.transition_scores_for(sequences)


class MockPretrainedModelProvider(PreTrainedModelProvider):
    def __init__(self, mock_outputs: Iterator[MockModelOutput], model_stats: ModelTrainingStats):
        self._mock_outputs = mock_outputs
        self._model_stats = model_stats

    def create_model_for_training(
        self, model_name: str, model_config: PretrainedConfig, device_map: dict[str, int]
    ) -> PreTrainedModel:
        return StandInSeq2SeqModel.create(NoGeneration(), self._model_stats)

    def create_model_for_inference(self, model_name: str) -> PreTrainedModel:
        recorded = RecordedGeneration(self._prepare_outputs(next(self._mock_outputs)))
        return StandInSeq2SeqModel.create(recorded, self._model_stats)

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
        self,
        model_settings: ModelSettings,
        model_name: ModelName,
        pretrained_tokenizer: PretrainedTokenizer,
        languages: ExperimentLanguages,
        mixed_precision: bool = False,
    ) -> PreTrainedModelProvider:
        return MockPretrainedModelProvider(iter(self._mock_outputs), self._model_stats)


# The sentence that the mock model below "translates" every source sentence to. Using a single
# fixed translation means that the smoke tests for the test and translate steps know the exact
# content of every generated file without having to store recorded model output in the repository.
MOCK_TRANSLATION = "Este es un borrador simulado."

# The log probabilities that the mock model reports for every generated token and sequence.
# They are the values that end up in the confidence files and in the confidence scores. Each
# sentence gets a slightly lower sequence score than the previous one, so that the sentences have
# distinct confidence scores, as they would with a real model.
MOCK_TOKEN_LOG_PROB = -0.5
MOCK_SEQUENCE_LOG_PROB = -0.25
MOCK_SEQUENCE_LOG_PROB_STEP = -0.01


def mock_sequence_log_prob(sentence_index: int) -> float:
    """The sequence score that the mock model reports for the nth sentence that it translates."""
    return MOCK_SEQUENCE_LOG_PROB + sentence_index * MOCK_SEQUENCE_LOG_PROB_STEP


class TranslationTokenIds:
    def __init__(
        self, pretrained_tokenizer: PretrainedTokenizer, languages: ExperimentLanguages, translation: str
    ) -> None:
        self._pretrained_tokenizer = pretrained_tokenizer
        self._languages = languages
        self._translation = translation
        self._token_ids: Optional[List[int]] = None

    def resolve(self) -> List[int]:
        # Resolved on first use, because the experiment's tokenizer is not written to the
        # experiment directory until the preprocess step runs.
        if self._token_ids is None:
            tokenizer = self._pretrained_tokenizer.load()
            trg_lang = self._languages.test_target() or self._languages.validation_target()
            translation_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self._translation))
            # A generated sequence starts with the decoder start token, followed by the forced
            # target language token, and ends with the end-of-sequence token.
            self._token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(trg_lang),
                *translation_token_ids,
                tokenizer.eos_token_id,
            ]
        return self._token_ids


class FixedTranslationPretrainedModelProvider(PreTrainedModelProvider):
    def __init__(
        self,
        pretrained_tokenizer: PretrainedTokenizer,
        languages: ExperimentLanguages,
        translation: str,
        model_stats: ModelTrainingStats,
        inference_model_names: List[str],
    ):
        self._translation_token_ids = TranslationTokenIds(pretrained_tokenizer, languages, translation)
        self._model_stats = model_stats
        self._inference_model_names = inference_model_names

    def create_model_for_training(
        self, model_name: str, model_config: PretrainedConfig, device_map: dict[str, int]
    ) -> PreTrainedModel:
        # Training ends with an evaluation pass, which generates because predict_with_generate is set.
        return StandInSeq2SeqModel.create(self._fixed_translation(), self._model_stats)

    def create_model_for_inference(self, model_name: str) -> PreTrainedModel:
        self._inference_model_names.append(model_name)
        return StandInSeq2SeqModel.create(self._fixed_translation(), self._model_stats)

    def _fixed_translation(self) -> FixedTranslationGeneration:
        return FixedTranslationGeneration(self._translation_token_ids)


class FixedTranslationPreTrainedModelProviderFactory(PreTrainedModelProviderFactory):
    """Creates mock models that always generate the same translation.

    Unlike MockPreTrainedModelProviderFactory, this factory does not need recorded model output,
    so it works with any amount of test data and any batch size.
    """

    def __init__(self, translation: str = MOCK_TRANSLATION, model_stats: ModelTrainingStats | None = None):
        self._translation = translation
        self._model_stats = model_stats or ModelTrainingStats()
        self._inference_model_names: List[str] = []

    @property
    def translation(self) -> str:
        return self._translation

    @property
    def stats(self) -> ModelTrainingStats:
        return self._model_stats

    @property
    def inference_model_names(self) -> List[str]:
        """The model name, i.e. the checkpoint path, that each inference model was created from."""
        return self._inference_model_names

    def create_pretrained_model_provider(
        self,
        model_settings: ModelSettings,
        model_name: ModelName,
        pretrained_tokenizer: PretrainedTokenizer,
        languages: ExperimentLanguages,
        mixed_precision: bool = False,
    ) -> PreTrainedModelProvider:
        return FixedTranslationPretrainedModelProvider(
            pretrained_tokenizer, languages, self._translation, self._model_stats, self._inference_model_names
        )
