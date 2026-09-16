from pathlib import Path

import pytest

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
def dirs(tmp_path):
    exp_dir = tmp_path / "exp"
    assets_dir = tmp_path / "assets"
    parent_dir = tmp_path / "parent"
    for directory in (exp_dir, assets_dir, parent_dir):
        directory.mkdir()
    return exp_dir, assets_dir, parent_dir


def copy_tokenizer(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        (destination / path.name).write_bytes(path.read_bytes())


def tokenizer_for(dirs, settings: TokenizerSettings, model: str = TINY_MODEL, parent: bool = False):
    exp_dir, assets_dir, parent_dir = dirs
    source = TokenizerSource(exp_dir, assets_dir, parent_dir if parent else None, model, settings)
    return PretrainedTokenizer(source, ModelName(model), exp_dir)


def test_the_model_is_loaded_when_the_experiment_has_no_tokenizer(dirs):
    loaded = tokenizer_for(dirs, TokenizerSettings({})).load()

    assert loaded.get_vocab()


def test_the_experiments_own_tokenizer_is_loaded_when_it_has_one(dirs, tiny_tokenizer_dir):
    exp_dir, _, _ = dirs
    copy_tokenizer(tiny_tokenizer_dir, exp_dir)

    assert tokenizer_for(dirs, TokenizerSettings({}), model="no/such-model").load().get_vocab()


def test_the_parent_tokenizer_is_loaded_when_the_experiment_has_none(dirs, tiny_tokenizer_dir):
    _, _, parent_dir = dirs
    copy_tokenizer(tiny_tokenizer_dir, parent_dir)

    assert tokenizer_for(dirs, TokenizerSettings({}), model="no/such-model", parent=True).load().get_vocab()


def test_the_tokenizer_is_loaded_once_and_shared(dirs):
    pretrained = tokenizer_for(dirs, TokenizerSettings({}))

    assert pretrained.load() is pretrained.load()


def test_building_and_loading_share_the_one_tokenizer(dirs):
    pretrained = tokenizer_for(dirs, TokenizerSettings({}))

    assert pretrained.build() is pretrained.load()


def test_the_tokenizer_assets_are_built_from_when_the_vocabulary_is_extended(dirs, tiny_tokenizer_dir):
    _, assets_dir, _ = dirs
    copy_tokenizer(tiny_tokenizer_dir, assets_dir)

    built = tokenizer_for(dirs, TokenizerSettings({"update_src": True}), model="no/such-model").build()

    assert built.get_vocab()


def test_reloading_takes_the_tokenizer_the_experiment_now_holds(dirs, tiny_tokenizer_dir):
    exp_dir, _, _ = dirs
    pretrained = tokenizer_for(dirs, TokenizerSettings({}))
    first = pretrained.load()
    copy_tokenizer(tiny_tokenizer_dir, exp_dir)

    reloaded = pretrained.reload_from_experiment()
    assert reloaded is not first
    assert pretrained.load() is reloaded


def test_padding_a_fast_tokenizer_is_not_warned_about_twice(dirs):
    # The warning fires once per tokenizer otherwise, on every batch of a long training run.
    loaded = tokenizer_for(dirs, TokenizerSettings({})).load()

    assert loaded.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] is True


def test_a_model_outside_the_known_families_cannot_convert_its_sentence_piece_model(dirs):
    # Only NLLB and MADLAD know how to be built from a bare SentencePiece model; anything else
    # reaches the end of the branch with no tokenizer at all.
    exp_dir, _, _ = dirs
    (exp_dir / "spiece.model").write_bytes(b"")

    with pytest.raises(AttributeError):
        tokenizer_for(dirs, TokenizerSettings({"update_src": True}), model="some/other-model").build()
