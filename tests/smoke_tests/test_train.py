import json
from pathlib import Path

import yaml

from silnlp.common.utils import get_git_revision_hash
from silnlp.nmt.config_utils import load_config
from tests.smoke_tests.mock_pretrained_model import FixedTranslationPreTrainedModelProviderFactory, ModelTrainingStats
from tests.smoke_tests.smoke_test_utils import (
    TRAIN_OUTPUT_PATTERNS,
    count_lines,
    create_model_with_mock_pretrained_model,
    delete_generated_paths,
    set_up_environment,
)

EXPERIMENT_NAME = "test_train"

# The training settings that are configured in the experiment's config.yml
MAX_STEPS = 2
TRAIN_BATCH_SIZE = 4


def test_train_creates_checkpoint():
    # The train step is run against the training and validation data that are stored in the
    # experiment directory, instead of running the preprocess step first.
    environment = set_up_environment()
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    delete_generated_paths(exp_dir, TRAIN_OUTPUT_PATTERNS)

    # The pretrained model is replaced with a mock, so that no model is downloaded and the
    # forward passes are done by a tiny randomly initialized model.
    config = load_config(EXPERIMENT_NAME, environment)
    model_provider_factory = FixedTranslationPreTrainedModelProviderFactory()
    model = create_model_with_mock_pretrained_model(config, model_provider_factory)
    config.set_seed()
    model.save_effective_config(config.exp_dir / f"effective-config-{get_git_revision_hash()}.yml")
    model.train()

    check_training_step(model_provider_factory.stats)
    check_training_data_was_used(exp_dir)
    check_effective_config(exp_dir)
    check_checkpoint(exp_dir)

    delete_generated_paths(exp_dir, TRAIN_OUTPUT_PATTERNS)


def check_training_step(model_stats: ModelTrainingStats):
    # There is one forward call per training step, because the gradient accumulation is configured
    # to be a single batch per step.
    assert model_stats.num_forward_calls == MAX_STEPS
    assert model_stats.observed_training_batch_sizes == [TRAIN_BATCH_SIZE] * MAX_STEPS
    assert model_stats.total_number_of_training_data_elements > 0


def check_training_data_was_used(exp_dir: Path):
    # The trainer reports how many training examples it loaded, which is the number of lines in the
    # training data that is stored in the experiment directory.
    with (exp_dir / "run" / "train_results.json").open("r", encoding="utf-8") as file:
        train_results = json.load(file)
    assert train_results["train_samples"] == count_lines(exp_dir / "train.src.txt")


def check_effective_config(exp_dir: Path):
    effective_config_paths = list(exp_dir.glob("effective-config-*.yml"))
    assert len(effective_config_paths) == 1

    with effective_config_paths[0].open("r", encoding="utf-8") as file:
        effective_config = yaml.safe_load(file)

    # The effective config records the model that the experiment is configured to train,
    # the real NLLB model instead of the tiny version that is actually used
    assert effective_config["model"] == "facebook/nllb-200-distilled-1.3B"
    assert effective_config["train"]["max_steps"] == MAX_STEPS
    assert effective_config["train"]["per_device_train_batch_size"] == TRAIN_BATCH_SIZE
    assert effective_config["train"]["output_dir"] == str(exp_dir / "run")
    assert effective_config["params"]["max_grad_norm"] == 1.0


def check_checkpoint(exp_dir: Path):
    model_dir = exp_dir / "run"

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
