import os
from pathlib import Path
from types import SimpleNamespace

import torch

os.environ.setdefault("SIL_NLP_DATA_PATH", str(Path(__file__).resolve().parents[1]))

from silnlp.nmt.hugging_face_config import (
    HuggingFaceNMTModel,
    PartialWordConstraintLogitsProcessor,
    SilTranslationPipeline,
)

# Approximately log(0.98): used to represent a high-confidence suggestion token.
HIGH_CONFIDENCE_LOG_PROB = -0.0202
# Approximately log(0.2): used to represent a low-confidence suggestion token.
LOW_CONFIDENCE_LOG_PROB = -1.6094


class FakeConstraintTokenizer:
    eos_token_id = 99
    all_special_ids = [99]

    def __init__(self) -> None:
        self._token_texts = {
            0: "cra",
            1: "crab",
            2: "crass",
            3: " ",
            99: "",
        }

    def get_vocab(self) -> dict[str, int]:
        return {f"token_{token_id}": token_id for token_id in self._token_texts}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._token_texts[token_id]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self._token_texts[token_id] for token_id in token_ids)


class FakeSuggestionTokenizer:
    eos_token_id = 99
    all_special_ids = [99]

    def __init__(self) -> None:
        self._char_to_id = {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        self._id_to_char = {token_id: ch for ch, token_id in self._char_to_id.items()}

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self._char_to_id)
        vocab["<eos>"] = self.eos_token_id
        return vocab

    def __call__(self, text_target=None, add_special_tokens=True, return_tensors=None, **kwargs):
        del kwargs
        if text_target is None:
            raise ValueError("text_target is required")
        ids = [self._char_to_id[ch] for ch in text_target]
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id == self.eos_token_id:
            return ""
        return self._id_to_char[token_id]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) -> str:
        del clean_up_tokenization_spaces
        text = []
        for token_id in token_ids:
            if token_id == self.eos_token_id:
                if skip_special_tokens:
                    continue
            elif token_id == 0 and skip_special_tokens:
                continue
            elif token_id in self._id_to_char:
                text.append(self._id_to_char[token_id])
        return "".join(text)


class FakeProviderFactory:
    def create_pretrained_model_provider(self, config, mixed_precision=False):
        del config, mixed_precision
        return object()


class FakeConfig:
    def __init__(self) -> None:
        self.data = {"seed": 0, "lang_codes": {}}
        self.infer = {}
        self.params = {"generation_num_beams": 1}
        self.model_prefix = ""


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(decoder_start_token_id=0, min_length=0, max_length=10)
        self.generation_config = SimpleNamespace(decoder_start_token_id=0, min_length=0, max_length=10)
        self.device = torch.device("cpu")


def test_partial_word_constraint_processor_allows_alternative_completions() -> None:
    tokenizer = FakeConstraintTokenizer()
    constraint = PartialWordConstraintLogitsProcessor(tokenizer, prompt_length=0, partial_word="cra")

    input_ids = torch.zeros((1, 0), dtype=torch.long)
    scores = torch.zeros((1, len(tokenizer.get_vocab())), dtype=torch.float32)
    constrained_scores = constraint(input_ids, scores)
    allowed_token_ids = torch.isfinite(constrained_scores[0]).nonzero().flatten().tolist()

    assert 1 in allowed_token_ids
    assert 2 in allowed_token_ids


def test_translation_pipeline_forward_pads_prompt_token_scores() -> None:
    transition_scores = torch.tensor([[0.7, 0.6]], dtype=torch.float32)
    generated_sequences = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)

    fake_model = SimpleNamespace(
        config=SimpleNamespace(min_length=0, max_length=10),
        generation_config=SimpleNamespace(min_length=0, max_length=10),
        generate=lambda **kwargs: SimpleNamespace(
            sequences=generated_sequences,
            scores=(torch.zeros(1, 1),),
            sequences_scores=torch.tensor([0.0]),
        ),
        compute_transition_scores=lambda *args, **kwargs: transition_scores,
    )

    pipeline = object.__new__(SilTranslationPipeline)
    pipeline.model = fake_model
    pipeline.check_inputs = lambda *args, **kwargs: None

    output = pipeline._forward(
        {
            "input_ids": torch.tensor([[5, 6]], dtype=torch.long),
            "decoder_input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        }
    )

    assert output["scores"].shape == (1, 1, 5)
    assert torch.equal(output["scores"][0, 0, :3], torch.zeros(3))
    assert torch.equal(output["scores"][0, 0, 3:], transition_scores[0])


def test_suggestion_translation_returns_remaining_characters_for_partial_word(monkeypatch) -> None:
    tokenizer = FakeSuggestionTokenizer()
    model = _create_model()
    captured_kwargs: dict = {}
    created_pipeline_count = 0

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            nonlocal created_pipeline_count
            del args
            self.model = kwargs["model"]
            created_pipeline_count += 1

        def __call__(self, sentences, **kwargs):
            del sentences
            captured_kwargs.update(kwargs)
            return [
                {
                    "translation_text": "crab",
                    "translation_token_ids": [0, 3, 18, 1, 2, tokenizer.eos_token_id],
                    "token_scores": torch.tensor(
                        [0.0, HIGH_CONFIDENCE_LOG_PROB, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
                    ),
                }
            ]

    monkeypatch.setattr("silnlp.nmt.hugging_face_config.SilTranslationPipeline", FakePipeline)
    monkeypatch.setattr(
        model, "_get_inference_components", lambda *args, **kwargs: (FakeModel(), tokenizer, "en", "en")
    )

    suggester = model.create_translation_suggester("en", "en", confidence_threshold=0.95)
    suggestion = suggester.suggestion_translation("source", "cra")

    assert suggestion == "b"
    logits_processors = captured_kwargs.get("logits_processor")
    assert logits_processors is not None
    assert len(logits_processors) == 1
    assert isinstance(logits_processors[0], PartialWordConstraintLogitsProcessor)
    assert captured_kwargs["num_beams"] == 1
    assert created_pipeline_count == 1


def test_suggestion_translation_returns_none_for_low_confidence_next_word(monkeypatch) -> None:
    tokenizer = FakeSuggestionTokenizer()
    model = _create_model()
    captured_kwargs: dict = {}

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            del args
            self.model = kwargs["model"]

        def __call__(self, sentences, **kwargs):
            del sentences
            captured_kwargs.update(kwargs)
            return [
                {
                    "translation_text": "hello world",
                    "translation_token_ids": [0] + _encode(tokenizer, "hello world") + [tokenizer.eos_token_id],
                    "token_scores": torch.tensor([0.0] * 7 + [LOW_CONFIDENCE_LOG_PROB] * 5, dtype=torch.float32),
                }
            ]

    monkeypatch.setattr("silnlp.nmt.hugging_face_config.SilTranslationPipeline", FakePipeline)
    monkeypatch.setattr(
        model, "_get_inference_components", lambda *args, **kwargs: (FakeModel(), tokenizer, "en", "en")
    )

    suggester = model.create_translation_suggester("en", "en", confidence_threshold=0.5)
    suggestion = suggester.suggestion_translation("source", "hello ")

    assert suggestion is None
    logits_processors = captured_kwargs.get("logits_processor")
    assert logits_processors is not None
    assert len(logits_processors) == 1
    assert isinstance(logits_processors[0], PartialWordConstraintLogitsProcessor)


def test_split_partial_translation_keeps_leading_space_with_partial_word() -> None:
    model = _create_model()
    tokenizer = FakeSuggestionTokenizer()

    prefix, partial_word = model._split_partial_translation(tokenizer, "hello wor")

    assert prefix == "hello"
    assert partial_word == " wor"


def test_split_partial_translation_treats_trailing_space_as_partial_word() -> None:
    model = _create_model()
    tokenizer = FakeSuggestionTokenizer()

    prefix, partial_word = model._split_partial_translation(tokenizer, "hello ")

    assert prefix == "hello"
    assert partial_word == " "


def _build_suggestion_inputs(
    tokenizer: FakeSuggestionTokenizer, generated_text: str, confident_words: set[int]
) -> tuple[list[int], torch.Tensor]:
    # Build token ids/scores for ``generated_text`` where every token belonging to a word
    # whose (space-separated) index is in ``confident_words`` is high-confidence and the
    # rest are low-confidence. The decoder start token and EOS are appended like the real
    # pipeline emits them.
    token_ids = [0] + _encode(tokenizer, generated_text) + [tokenizer.eos_token_id]
    scores = [0.0]  # decoder start token
    word_index = 0
    for ch in generated_text:
        if ch == " ":
            word_index += 1
            scores.append(HIGH_CONFIDENCE_LOG_PROB)
        else:
            scores.append(HIGH_CONFIDENCE_LOG_PROB if word_index in confident_words else LOW_CONFIDENCE_LOG_PROB)
    scores.append(0.0)  # EOS
    return token_ids, torch.tensor(scores, dtype=torch.float32)


def test_extract_suggestion_text_returns_full_confident_phrase() -> None:
    model = _create_model()
    tokenizer = FakeSuggestionTokenizer()
    token_ids, scores = _build_suggestion_inputs(tokenizer, "go home now", confident_words={0, 1, 2})

    suggestion = model._extract_suggestion_text(tokenizer, "go", token_ids, scores, confidence_threshold=0.9)

    assert suggestion == " home now"


def test_extract_suggestion_text_trims_low_confidence_trailing_word() -> None:
    model = _create_model()
    tokenizer = FakeSuggestionTokenizer()
    # The final word "now" is low-confidence, so it should be dropped at its word boundary.
    token_ids, scores = _build_suggestion_inputs(tokenizer, "go home now", confident_words={0, 1})

    suggestion = model._extract_suggestion_text(tokenizer, "go", token_ids, scores, confidence_threshold=0.9)

    assert suggestion == " home"


def test_extract_suggestion_text_drops_word_truncated_without_eos() -> None:
    model = _create_model()
    tokenizer = FakeSuggestionTokenizer()
    # Generation stops mid-word ("ho") with no EOS, so the incomplete word is not suggested.
    token_ids = [0] + _encode(tokenizer, "go ho")
    scores = torch.tensor([0.0] + [HIGH_CONFIDENCE_LOG_PROB] * 5, dtype=torch.float32)

    suggestion = model._extract_suggestion_text(tokenizer, "go", token_ids, scores, confidence_threshold=0.9)

    assert suggestion is None


def _create_model() -> HuggingFaceNMTModel:
    return HuggingFaceNMTModel(
        FakeConfig(), mixed_precision=False, num_devices=1, pretrained_model_provider_factory=FakeProviderFactory()
    )


def _encode(tokenizer: FakeSuggestionTokenizer, text: str) -> list[int]:
    return tokenizer(text_target=text, add_special_tokens=False)["input_ids"]
