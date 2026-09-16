from silnlp.nmt.config_keys import RenamedConfigKeys


def test_a_renamed_key_is_warned_about(caplog):
    RenamedConfigKeys.of_llm().warn_about({"train": {"group_by_length": True}})

    assert "train.group_by_length was renamed to train.train_sampling_strategy" in caplog.text


def test_a_current_key_is_not_warned_about(caplog):
    RenamedConfigKeys.of_llm().warn_about({"train": {"train_sampling_strategy": "group_by_length"}})

    assert caplog.text == ""


def test_a_section_that_is_not_a_mapping_is_passed_over(caplog):
    RenamedConfigKeys.of_llm().warn_about({"train": "not a section"})

    assert caplog.text == ""


def test_a_missing_section_is_passed_over(caplog):
    RenamedConfigKeys.of_llm().warn_about({})

    assert caplog.text == ""


def test_the_sequence_to_sequence_models_have_a_renamed_evaluation_key(caplog):
    RenamedConfigKeys.of_seq2seq().warn_about({"eval": {"include_inputs_for_metrics": True}})

    assert "eval.include_inputs_for_metrics was renamed to eval.include_for_metrics" in caplog.text


def test_the_language_models_share_the_keys_renamed_for_every_model(caplog):
    RenamedConfigKeys.of_llm().warn_about({"params": {"warmup_ratio": 0.1}})

    assert "params.warmup_ratio was renamed to params.warmup_steps" in caplog.text
