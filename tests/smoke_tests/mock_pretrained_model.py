from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, cast
from unittest.mock import Mock, create_autospec

import torch
from transformers import AutoModelForSeq2SeqLM, PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

from silnlp.nmt.seq2seq_config import PreTrainedModelProvider, PreTrainedModelProviderFactory, Seq2SeqConfig

_TINY_MODEL_NAME = "hf-internal-testing/tiny-random-nllb"

# The function that stands in for PreTrainedModel.generate. It receives the arguments that the
# translation pipeline passes to the model, which includes the batch's "input_ids".
MockGenerate = Callable[..., GenerateBeamEncoderDecoderOutput]


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


def _generation_not_expected(*args, **kwargs) -> GenerateBeamEncoderDecoderOutput:
    raise AssertionError("The model was asked to generate translations, but no mock output was provided.")


def create_mock_pretrained_model(
    generate: MockGenerate = _generation_not_expected, model_stats: ModelTrainingStats | None = None
) -> PreTrainedModel:
    if model_stats is None:
        model_stats = ModelTrainingStats()

    underlying_model = cast(PreTrainedModel, AutoModelForSeq2SeqLM.from_pretrained(_TINY_MODEL_NAME, token=False))
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
        output = generate(*args, **kwargs)

        # SilTranslator calls compute_transition_scores and requires a shape that
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
        return create_mock_pretrained_model(model_stats=self._model_stats)

    def create_model_for_inference(self, model_name: str) -> PreTrainedModel:
        recorded_outputs = self._prepare_outputs(next(self._mock_outputs))
        return create_mock_pretrained_model(lambda *args, **kwargs: next(recorded_outputs), self._model_stats)

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


def create_fixed_translation_mock_pretrained_model(
    get_translation_token_ids: Callable[[], List[int]], model_stats: ModelTrainingStats
) -> PreTrainedModel:
    """Create a mock model that generates the same translation for every sentence in every batch."""
    num_translated_sentences = 0

    def generate(*args, **kwargs) -> GenerateBeamEncoderDecoderOutput:
        nonlocal num_translated_sentences
        input_ids: Optional[torch.Tensor] = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        assert input_ids is not None
        batch_size = input_ids.shape[0]
        device = input_ids.device

        sequences = torch.tensor(get_translation_token_ids(), dtype=torch.long, device=device).repeat(batch_size, 1)
        scores = torch.full(tuple(sequences.shape), MOCK_TOKEN_LOG_PROB, dtype=torch.float32, device=device)
        sequences_scores = torch.tensor(
            [mock_sequence_log_prob(num_translated_sentences + i) for i in range(batch_size)],
            dtype=torch.float32,
            device=device,
        )
        num_translated_sentences += batch_size
        return GenerateBeamEncoderDecoderOutput(
            sequences=cast(torch.LongTensor, sequences),
            beam_indices=cast(torch.LongTensor, torch.zeros_like(sequences)),
            scores=(cast(torch.FloatTensor, scores),),
            sequences_scores=cast(torch.FloatTensor, sequences_scores),
        )

    return create_mock_pretrained_model(generate, model_stats)


class FixedTranslationPretrainedModelProvider(PreTrainedModelProvider):
    def __init__(
        self,
        config: Seq2SeqConfig,
        translation: str,
        model_stats: ModelTrainingStats,
        inference_model_names: List[str],
    ):
        self._config = config
        self._translation = translation
        self._model_stats = model_stats
        self._inference_model_names = inference_model_names
        self._translation_token_ids: Optional[List[int]] = None

    def create_model_for_training(
        self, model_name: str, model_config: PretrainedConfig, device_map: dict[str, int]
    ) -> PreTrainedModel:
        return create_mock_pretrained_model(model_stats=self._model_stats)

    def create_model_for_inference(self, model_name: str) -> PreTrainedModel:
        self._inference_model_names.append(model_name)
        return create_fixed_translation_mock_pretrained_model(self._get_translation_token_ids, self._model_stats)

    def _get_translation_token_ids(self) -> List[int]:
        # The ids are resolved when the first batch is generated, because the experiment's
        # tokenizer is not written to the experiment directory until the preprocess step runs.
        if self._translation_token_ids is None:
            tokenizer = self._config.get_tokenizer()
            trg_lang = self._config.test_trg_lang or self._config.val_trg_lang
            translation_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self._translation))
            # A generated sequence starts with the decoder start token, followed by the forced
            # target language token, and ends with the end-of-sequence token.
            self._translation_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(trg_lang),
                *translation_token_ids,
                tokenizer.eos_token_id,
            ]
        return self._translation_token_ids


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
        self, config: Seq2SeqConfig, mixed_precision: bool = False
    ) -> PreTrainedModelProvider:
        return FixedTranslationPretrainedModelProvider(
            config, self._translation, self._model_stats, self._inference_model_names
        )
