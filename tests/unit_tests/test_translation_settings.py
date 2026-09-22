from silnlp.nmt.translation_settings import CheckpointRetention, ModelSettings, TranslationSettings


def test_the_batch_size_is_the_configured_one():
    assert TranslationSettings({"infer_batch_size": 8}, {}).batch_size() == 8


def test_the_configured_number_of_beams_is_used():
    assert TranslationSettings({"num_beams": 4}, {"generation_num_beams": 9}).beam_count() == 4


def test_the_training_arguments_settle_the_beams_when_inference_names_none():
    assert TranslationSettings({}, {"generation_num_beams": 9}).beam_count() == 9


def test_no_beams_are_named_when_neither_section_sets_any():
    assert TranslationSettings({}, {}).beam_count() is None


def test_the_sampling_temperature_is_the_configured_one():
    assert TranslationSettings({"temperature": 0.75}, {}).temperature() == 0.75


def test_a_method_for_several_drafts_is_reported_as_configured():
    assert TranslationSettings({"multiple_translations_method": "hybrid"}, {}).multiple_translations_method() == "hybrid"


def test_the_model_is_built_with_the_configured_dropouts():
    settings = ModelSettings(
        {"dropout": 0.1, "attention_dropout": 0.2, "activation_dropout": 0.3, "attn_implementation": "sdpa"}
    )

    assert settings.dropout() == 0.1
    assert settings.attention_dropout() == 0.2
    assert settings.activation_dropout() == 0.3
    assert settings.attention_implementation() == "sdpa"


def test_checkpoints_keep_what_the_config_does_not_ask_to_delete():
    retention = CheckpointRetention(
        {"delete_checkpoint_optimizer_state": False, "delete_checkpoint_tokenizer": False}
    )

    assert retention.keeps_optimizer_state()
    assert retention.keeps_tokenizers()


def test_checkpoints_drop_what_the_config_asks_to_delete():
    retention = CheckpointRetention({"delete_checkpoint_optimizer_state": True, "delete_checkpoint_tokenizer": True})

    assert not retention.keeps_optimizer_state()
    assert not retention.keeps_tokenizers()
