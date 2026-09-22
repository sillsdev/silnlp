from pathlib import Path

import pytest

from silnlp.common.utils import Side
from silnlp.nmt.huggingface_tokenizer import HuggingFaceTokenizer
from silnlp.nmt.model_name import ModelName
from silnlp.nmt.pretrained_tokenizer import PretrainedTokenizer
from silnlp.nmt.tokenizer_settings import TokenizerSettings, TokenizerSource

TINY_MODEL = "hf-internal-testing/tiny-random-nllb"


@pytest.fixture(scope="module")
def tiny_tokenizer_dir(tmp_path_factory) -> Path:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    directory = tmp_path_factory.mktemp("tiny")
    AutoTokenizer.from_pretrained(TINY_MODEL, use_fast=True, token=False).save_pretrained(str(directory))
    return directory


@pytest.fixture
def exp_dir(tmp_path: Path, tiny_tokenizer_dir: Path) -> Path:
    directory = tmp_path / "exp"
    directory.mkdir()
    for path in tiny_tokenizer_dir.iterdir():
        (directory / path.name).write_bytes(path.read_bytes())
    return directory


@pytest.fixture
def pretrained(exp_dir: Path, tmp_path: Path) -> PretrainedTokenizer:
    source = TokenizerSource(exp_dir, tmp_path / "assets", None, TINY_MODEL, TokenizerSettings({}))
    return PretrainedTokenizer(source, ModelName(TINY_MODEL), exp_dir)


def tokenizer_for(pretrained: PretrainedTokenizer) -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer(pretrained, {"en": "eng_Latn"}, 200, 200)


def test_the_configured_language_code_is_used_for_the_source(pretrained):
    tokenizer = tokenizer_for(pretrained)
    tokenizer.set_src_lang("en")

    assert pretrained.load().src_lang == "eng_Latn"


def test_an_unconfigured_language_is_passed_through_as_written(pretrained):
    tokenizer = tokenizer_for(pretrained)
    tokenizer.set_src_lang("sdl")

    assert pretrained.load().src_lang == "sdl"


def test_special_tokens_are_left_out_of_the_detokenized_text(pretrained):
    tokenizer = tokenizer_for(pretrained)

    assert "</s>" not in tokenizer.detokenize("</s> hello")


def test_extending_the_vocabulary_is_picked_up_rather_than_leaving_a_version_behind(pretrained, exp_dir):
    # Extending the vocabulary replaces the underlying tokenizer, so anything holding the previous
    # instance would keep tokenizing against a vocabulary missing every token just added.
    tokenizer = tokenizer_for(pretrained)
    assert "<extended>" in tokenizer.detokenize("<extended> hello")

    extended = pretrained.load()
    extended.add_special_tokens({"extra_special_tokens": ["<extended>"]}, replace_extra_special_tokens=False)
    extended.save_pretrained(str(exp_dir))
    pretrained.reload_from_experiment()

    assert "<extended>" not in tokenizer.detokenize("<extended> hello")


def test_the_tokenizer_survives_being_made_before_the_vocabulary_is_settled(pretrained, exp_dir):
    tokenizer = tokenizer_for(pretrained)

    pretrained.reload_from_experiment()
    tokenizer.set_trg_lang("en")

    assert tokenizer.tokenize(Side.TARGET, "hello")
