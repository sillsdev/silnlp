from silnlp.nmt.experiment_settings import EvaluationSettings, ScoringSettings, TrainingSettings
from silnlp.nmt.model_name import ModelName


def evaluation_config() -> dict:
    return {
        "eval_strategy": "steps",
        "load_best_model_at_end": True,
        "early_stopping": 3,
        "metric_for_best_model": "bleu",
    }


def test_evaluation_is_left_alone_when_there_is_a_validation_split():
    config = evaluation_config()
    EvaluationSettings(config).disable_unless(has_validation_split=True)

    assert config == evaluation_config()


def test_evaluation_is_turned_off_entirely_without_a_validation_split():
    config = evaluation_config()
    EvaluationSettings(config).disable_unless(has_validation_split=False)

    assert config == {
        "eval_strategy": "no",
        "load_best_model_at_end": False,
        "early_stopping": None,
        "metric_for_best_model": None,
    }


def training_config(**overrides) -> dict:
    config = {
        "max_source_length": 200,
        "max_target_length": 200,
        "auto_grad_acc": False,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 4,
    }
    config.update(overrides)
    return config


def test_madlad_is_given_longer_sequences_than_the_default():
    config = training_config()
    TrainingSettings(config).fit_to(ModelName("google/madlad400-3b-mt"))

    assert config["max_source_length"] == 256
    assert config["max_target_length"] == 256


def test_other_models_keep_the_configured_sequence_lengths():
    config = training_config()
    TrainingSettings(config).fit_to(ModelName("facebook/nllb-200-distilled-1.3B"))

    assert config["max_source_length"] == 200


def test_automatic_gradient_accumulation_replaces_the_configured_batching():
    config = training_config(auto_grad_acc=True)
    TrainingSettings(config).fit_to(ModelName("facebook/nllb-200-distilled-1.3B"))

    assert config["per_device_train_batch_size"] == 64
    assert config["gradient_accumulation_steps"] == 1


def test_the_configured_batching_stands_when_it_is_not_automatic():
    config = training_config()
    TrainingSettings(config).fit_to(ModelName("facebook/nllb-200-distilled-1.3B"))

    assert config["per_device_train_batch_size"] == 16
    assert config["gradient_accumulation_steps"] == 4


def test_a_madlad_experiment_can_also_batch_automatically():
    config = training_config(auto_grad_acc=True)
    TrainingSettings(config).fit_to(ModelName("google/madlad400-3b-mt"))

    assert config["max_source_length"] == 256
    assert config["per_device_train_batch_size"] == 64


def test_sacrebleu_falls_back_to_its_own_default_tokenizer():
    assert ScoringSettings({}).sacrebleu_tokenizer() == "13a"


def test_a_configured_sacrebleu_tokenizer_is_used_instead():
    assert ScoringSettings({"sacrebleu_tokenize": "flores200"}).sacrebleu_tokenizer() == "flores200"
