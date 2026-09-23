import json
from pathlib import Path
from typing import Any, List, Optional

import pytest

from silnlp.nmt.checkpoints import CheckpointDirectory, CheckpointType
from silnlp.nmt.model_name import ModelName
from silnlp.nmt.pretrained_model_loader import PretrainedModelLoader
from silnlp.nmt.tokenizer_settings import TokenizerSettings
from silnlp.nmt.translation_settings import ModelSettings


class RecordingProvider:
    """Stands in for the provider, recording what it was asked to build a model from."""

    def __init__(self) -> None:
        self.training_calls: List[dict] = []
        self.inference_names: List[str] = []

    def create_model_for_training(self, model_name: str, model_config: Any, device_map: Optional[dict]) -> Any:
        self.training_calls.append({"model_name": model_name, "device_map": device_map})
        return StandInModel()

    def create_model_for_inference(self, model_name: str) -> Any:
        self.inference_names.append(model_name)
        return StandInModel()


class StandInConfig:
    decoder_start_token_id = 3


class StandInGenerationConfig:
    def __init__(self) -> None:
        self.max_length = None
        self.max_new_tokens = None
        self.decoder_start_token_id = None
        self.forced_bos_token_id = None


class StandInEmbeddings:
    class _Weight:
        def size(self, dim: int) -> int:
            return 1000

    weight = _Weight()


class StandInModel:
    def __init__(self) -> None:
        self.config = StandInConfig()
        self.generation_config = StandInGenerationConfig()

    def get_input_embeddings(self) -> StandInEmbeddings:
        return StandInEmbeddings()

    def resize_token_embeddings(self, size: int, pad_to_multiple_of: Optional[int]) -> None:
        raise AssertionError("the vocabulary fits, so the embeddings should be left alone")


class StandInTokenizer:
    pad_token_id = 0

    def __len__(self) -> int:
        return 500

    def convert_tokens_to_ids(self, token: str) -> int:
        return 7


class StandInPretrainedTokenizer:
    def load(self) -> StandInTokenizer:
        return StandInTokenizer()


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    # Not created: a run directory only appears once training has written a checkpoint.
    return tmp_path / "run"


def trained(model_dir: Path, step: int = 500) -> Path:
    (model_dir / f"checkpoint-{step}").mkdir(parents=True)
    return model_dir


def loader_for(
    model_dir: Path, provider: RecordingProvider, model: str = "facebook/nllb-200-distilled-1.3B", num_devices: int = 1
) -> PretrainedModelLoader:
    return PretrainedModelLoader(
        provider,
        model,
        ModelName(model),
        StandInPretrainedTokenizer(),
        TokenizerSettings({}),
        ModelSettings({"dropout": 0.1, "attention_dropout": 0.1, "activation_dropout": 0.0, "attn_implementation": "sdpa"}),
        CheckpointDirectory(model_dir),
        num_devices,
    )


def trainer_state(model_dir: Path, max_steps: int = 5000, best: Optional[str] = None) -> None:
    state = {"max_steps": max_steps}
    if best is not None:
        state["best_model_checkpoint"] = best
    (model_dir / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")


def test_the_base_model_is_used_when_nothing_has_been_trained(model_dir):
    provider = RecordingProvider()
    loader_for(model_dir, provider).for_inference(CheckpointType.LAST, "eng_Latn", "spa_Latn")

    assert provider.inference_names == ["facebook/nllb-200-distilled-1.3B"]


def test_a_checkpoint_is_used_once_the_model_has_been_trained(model_dir):
    trained(model_dir)
    trainer_state(model_dir, max_steps=500)
    provider = RecordingProvider()

    loader_for(model_dir, provider).for_inference(CheckpointType.LAST, "eng_Latn", "spa_Latn")

    assert provider.inference_names[0].endswith("checkpoint-500")


def test_an_inference_model_is_given_room_to_generate(model_dir):
    model = loader_for(model_dir, RecordingProvider()).for_inference(CheckpointType.LAST, "eng_Latn", "spa_Latn")

    assert model.generation_config.max_length == 512


def test_a_model_that_is_not_multilingual_has_no_language_forced_on_it(model_dir):
    # Only mBART, mBART-50, M2M100 and NLLB need the target language as the first generated token.
    model = loader_for(model_dir, RecordingProvider()).for_inference(CheckpointType.LAST, "eng_Latn", "spa_Latn")

    assert model.generation_config.forced_bos_token_id is None


def test_a_model_with_no_start_token_is_rejected(model_dir):
    class NoStartTokenModel(StandInModel):
        def __init__(self) -> None:
            super().__init__()
            self.config.decoder_start_token_id = None

    provider = RecordingProvider()
    provider.create_model_for_inference = lambda model_name: NoStartTokenModel()

    with pytest.raises(ValueError, match="decoder_start_token_id"):
        loader_for(model_dir, provider).for_inference(CheckpointType.LAST, "", "")


def test_madlad_generates_from_the_padding_token(model_dir):
    model = loader_for(model_dir, RecordingProvider(), model="google/madlad400-3b-mt").for_inference(
        CheckpointType.LAST, "en", "es"
    )

    assert model.config.decoder_start_token_id == 0
    assert model.generation_config.max_new_tokens == 256
