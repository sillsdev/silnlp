import json
from pathlib import Path

import yaml

from silnlp.nmt.seq2seq_config import Seq2SeqConfig
from tests.smoke_tests.mock_pretrained_model import FixedTranslationPreTrainedModelProviderFactory, ModelTrainingStats
from tests.smoke_tests.smoke_test_utils import (
    PREPROCESS_OUTPUT_PATTERNS,
    TRAIN_OUTPUT_PATTERNS,
    create_model_with_mock_pretrained_model,
    delete_generated_paths,
    load_experiment_config,
    run_preprocess_step,
    run_train_step,
    set_up_environment,
)

EXPERIMENT_NAME = "test_train"

# The training settings that are configured in the experiment's config.yml
MAX_STEPS = 2
TRAIN_BATCH_SIZE = 4


def test_train_creates_checkpoint():
    environment = set_up_environment()
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    delete_generated_paths(exp_dir, PREPROCESS_OUTPUT_PATTERNS + TRAIN_OUTPUT_PATTERNS)

    # The train step needs the data sets that the preprocess step creates.
    run_preprocess_step(load_experiment_config(environment, EXPERIMENT_NAME, Seq2SeqConfig))

    # The pretrained model is replaced with a mock, so that no model is downloaded and the
    # forward passes are done by a tiny randomly initialized model.
    config = load_experiment_config(environment, EXPERIMENT_NAME, Seq2SeqConfig)
    model_provider_factory = FixedTranslationPreTrainedModelProviderFactory()
    model = create_model_with_mock_pretrained_model(config, model_provider_factory)
    run_train_step(config, model)

    check_training_step(model_provider_factory.stats)
    check_effective_config(exp_dir)
    check_checkpoint(exp_dir)

    delete_generated_paths(exp_dir, PREPROCESS_OUTPUT_PATTERNS + TRAIN_OUTPUT_PATTERNS)


def check_training_step(model_stats: ModelTrainingStats):
    # There is one forward call per training step, because the gradient accumulation is configured
    # to be a single batch per step.
    assert model_stats.num_forward_calls == MAX_STEPS
    assert model_stats.observed_training_batch_sizes == [TRAIN_BATCH_SIZE] * MAX_STEPS
    assert model_stats.total_number_of_training_data_elements > 0


def check_effective_config(exp_dir: Path):
    effective_config_paths = list(exp_dir.glob("effective-config-*.yml"))
    assert len(effective_config_paths) == 1

    with effective_config_paths[0].open("r", encoding="utf-8") as file:
        effective_config = yaml.safe_load(file)
    assert effective_config["model"] == "facebook/nllb-200-distilled-1.3B"
    assert effective_config["train"]["max_steps"] == MAX_STEPS
    assert effective_config["train"]["per_device_train_batch_size"] == TRAIN_BATCH_SIZE
    assert effective_config["train"]["output_dir"] == str(exp_dir / "run")
    # The effective config also contains the training arguments that the experiment's config file
    # does not set, e.g. the defaults of the Huggingface trainer
    assert effective_config["params"]["max_grad_norm"] == 1.0


def check_checkpoint(exp_dir: Path):
    model_dir = exp_dir / "run"
    assert (model_dir / "train_results.json").is_file()

    with (model_dir / "trainer_state.json").open("r", encoding="utf-8") as file:
        trainer_state = json.load(file)
    assert trainer_state["global_step"] == MAX_STEPS
    assert trainer_state["max_steps"] == MAX_STEPS

    checkpoint_dir = model_dir / f"checkpoint-{MAX_STEPS}"
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "model.safetensors").is_file()
    assert (checkpoint_dir / "config.json").is_file()

    # The optimizer state and the tokenizer are deleted from the checkpoints after training,
    # because they are not needed for inference.
    assert not (checkpoint_dir / "optimizer.pt").exists()
    assert not (checkpoint_dir / "tokenizer.json").exists()
