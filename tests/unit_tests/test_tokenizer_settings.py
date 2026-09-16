import pytest

from silnlp.nmt.tokenizer_settings import TokenizerSettings, TokenizerSource


def settings(**overrides) -> TokenizerSettings:
    return TokenizerSettings(overrides)


def test_a_missing_tokenizer_section_updates_nothing():
    assert not TokenizerSettings(None).updates_either()
    assert not TokenizerSettings({}).updates_either()


def test_either_side_asking_for_an_update_is_enough():
    assert settings(update_src=True, update_trg=False).updates_either()
    assert settings(update_src=False, update_trg=True).updates_either()
    assert not settings(update_src=False, update_trg=False).updates_either()


def test_both_sides_are_updated_only_when_each_asks():
    assert settings(update_src=True, update_trg=True).updates_both()
    assert not settings(update_src=True, update_trg=False).updates_both()


def test_a_shared_vocabulary_is_the_two_sizes_together():
    assert settings(src_vocab_size=500, trg_vocab_size=300).shared_vocab_size() == 800


def test_the_flags_default_to_off_when_absent():
    assert not settings().shares_vocab()
    assert not settings().trains_tokens()


@pytest.fixture
def dirs(tmp_path):
    exp_dir = tmp_path / "exp"
    assets_dir = tmp_path / "assets"
    parent_dir = tmp_path / "parent"
    for directory in (exp_dir, assets_dir, parent_dir):
        directory.mkdir()
    return exp_dir, assets_dir, parent_dir


def source_for(dirs, settings: TokenizerSettings, parent: bool = False) -> TokenizerSource:
    exp_dir, assets_dir, parent_dir = dirs
    return TokenizerSource(exp_dir, assets_dir, parent_dir if parent else None, "facebook/nllb-200", settings)


def config_in(directory):
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_the_model_is_used_when_nothing_else_is_available(dirs):
    assert source_for(dirs, settings(update_src=True)).path_for_building() == "facebook/nllb-200"


def test_the_parent_experiment_is_preferred_over_the_model(dirs):
    exp_dir, _, parent_dir = dirs
    assert source_for(dirs, settings(update_src=True), parent=True).path_for_building() == str(parent_dir)


def test_the_tokenizer_assets_are_used_when_the_vocabulary_is_being_extended(dirs):
    exp_dir, assets_dir, _ = dirs
    config_in(assets_dir)

    assert source_for(dirs, settings(update_src=True), parent=True).path_for_building() == str(assets_dir)


def test_the_tokenizer_assets_are_ignored_when_the_vocabulary_is_not_being_extended(dirs):
    exp_dir, assets_dir, parent_dir = dirs
    config_in(assets_dir)

    assert source_for(dirs, settings(update_src=False), parent=True).path_for_building() == str(parent_dir)


def test_the_experiment_is_used_when_it_already_has_a_tokenizer_and_none_is_being_built(dirs):
    exp_dir, _, _ = dirs
    config_in(exp_dir)

    assert source_for(dirs, settings(update_src=False), parent=True).path_for_building() == str(exp_dir)


def test_the_experiment_is_passed_over_while_the_vocabulary_is_being_extended(dirs):
    # An extended vocabulary starts from the assets or the parent, not from a previous run's tokenizer.
    exp_dir, _, parent_dir = dirs
    config_in(exp_dir)

    assert source_for(dirs, settings(update_src=True), parent=True).path_for_building() == str(parent_dir)


def test_loading_prefers_the_experiments_own_tokenizer(dirs):
    exp_dir, _, _ = dirs
    config_in(exp_dir)

    assert source_for(dirs, settings(update_src=True), parent=True).path_for_loading() == str(exp_dir)


def test_loading_falls_back_to_the_parent_then_the_model(dirs):
    _, _, parent_dir = dirs

    assert source_for(dirs, settings(), parent=True).path_for_loading() == str(parent_dir)
    assert source_for(dirs, settings()).path_for_loading() == "facebook/nllb-200"


@pytest.mark.parametrize("name", ["sentencepiece.bpe.model", "spiece.model"])
def test_a_sentence_piece_model_with_no_tokenizer_beside_it_still_needs_converting(name, dirs):
    exp_dir, _, _ = dirs
    (exp_dir / name).write_text("", encoding="utf-8")

    assert source_for(dirs, settings(update_src=True)).holds_unconverted_sentence_piece_model()


def test_a_sentence_piece_model_that_has_already_been_converted_is_left_alone(dirs):
    exp_dir, _, _ = dirs
    (exp_dir / "spiece.model").write_text("", encoding="utf-8")
    config_in(exp_dir)

    assert not source_for(dirs, settings(update_src=True)).holds_unconverted_sentence_piece_model()


def test_a_sentence_piece_model_is_ignored_when_the_vocabulary_is_not_being_extended(dirs):
    exp_dir, _, _ = dirs
    (exp_dir / "spiece.model").write_text("", encoding="utf-8")

    assert not source_for(dirs, settings(update_src=False)).holds_unconverted_sentence_piece_model()
