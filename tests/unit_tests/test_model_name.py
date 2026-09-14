from pathlib import Path

import pytest

from silnlp.nmt.model_name import ModelName


@pytest.mark.parametrize(
    "name",
    ["facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-distilled-600M", "facebook/nllb-200-3.3B"],
)
def test_nllb_models(name):
    model_name = ModelName(name)
    assert model_name.is_nllb()
    assert not model_name.is_madlad()
    assert not model_name.is_t5()
    assert not model_name.looks_decoder_only()


@pytest.mark.parametrize("name", ["google/madlad400-3b-mt", "google/madlad400-10b-mt"])
def test_madlad_models_are_t5_models(name):
    model_name = ModelName(name)
    assert model_name.is_madlad()
    assert model_name.is_t5()
    assert not model_name.is_nllb()


@pytest.mark.parametrize(
    "name",
    [
        "google/gemma-2-2b-it",
        "google/translate-gemma-2b",
        "google/translategemma-2b",
        "tencent/Hunyuan-MT-7B",
        "Hunyuan-MT-7B",
    ],
)
def test_decoder_only_models(name):
    assert ModelName(name).looks_decoder_only()


@pytest.mark.parametrize("name", ["facebook/nllb-200-distilled-1.3B", "google/madlad400-3b-mt", "facebook/m2m100_418M"])
def test_seq2seq_models_are_not_decoder_only(name):
    assert not ModelName(name).looks_decoder_only()


@pytest.mark.parametrize("name", ["google/translate-gemma-2b", "google/translategemma-2b"])
def test_translate_gemma_uses_its_own_chat_template(name):
    assert ModelName(name).uses_translate_gemma_template()


@pytest.mark.parametrize("name", ["google/gemma-2-2b-it", "tencent/Hunyuan-MT-7B"])
def test_other_decoder_only_models_use_the_configured_template(name):
    assert not ModelName(name).uses_translate_gemma_template()


def test_the_translate_gemma_check_ignores_case_but_the_decoder_only_check_does_not():
    model_name = ModelName("Google/Translate-Gemma-2b")
    assert model_name.uses_translate_gemma_template()
    assert not model_name.looks_decoder_only()


def test_an_unrecognized_model_matches_nothing():
    model_name = ModelName("facebook/m2m100_418M")
    assert not model_name.is_nllb()
    assert not model_name.is_madlad()
    assert not model_name.is_t5()
    assert not model_name.looks_decoder_only()


def test_models_of_the_same_family_match_across_sizes():
    assert ModelName("facebook/nllb-200-distilled-600M").same_family_as(ModelName("facebook/nllb-200-3.3B"))


def test_models_of_different_families_do_not_match():
    assert not ModelName("facebook/nllb-200-3.3B").same_family_as(ModelName("google/madlad400-3b-mt"))


def test_two_unrecognized_models_count_as_the_same_family():
    # Both fall back to the empty family, which is what the parent-model check compares.
    assert ModelName("facebook/m2m100_418M").same_family_as(ModelName("t5-small"))


def test_the_tokenizer_assets_directory_is_named_after_the_family():
    assets_dir = Path("/assets")
    assert ModelName("facebook/nllb-200-3.3B").tokenizer_assets_dir(assets_dir) == (
        assets_dir / "tokenizers" / "facebook/nllb-200"
    )
    assert ModelName("google/madlad400-3b-mt").tokenizer_assets_dir(assets_dir) == (
        assets_dir / "tokenizers" / "google/madlad400"
    )


def test_rendering_a_model_name_gives_back_what_was_configured():
    assert str(ModelName("facebook/nllb-200-3.3B")) == "facebook/nllb-200-3.3B"


def test_nllb_trains_a_byte_pair_vocabulary():
    settings = ModelName("facebook/nllb-200-3.3B").sentence_piece_settings()
    assert settings["type"] == "BPE"
    assert settings["special_tokens"] == ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def test_madlad_trains_a_unigram_vocabulary():
    settings = ModelName("google/madlad400-3b-mt").sentence_piece_settings()
    assert settings["type"] == "Unigram"
    assert settings["special_tokens"] == ["<unk>", "<s>", "</s>"]
    assert settings["unk_token"] == "<unk>"


def test_a_model_with_no_known_family_has_no_sentence_piece_settings():
    with pytest.raises(KeyError):
        ModelName("facebook/m2m100_418M").sentence_piece_settings()
