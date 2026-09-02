"""Shared helpers for the smoke tests of the NMT pipeline.

The step tests, i.e. test_preprocess, test_train, test_test and test_translate, make the same
calls that the main function of the corresponding command makes. The experiment tests, i.e.
test_experiment and test_experiment_llm, run all of the steps as one experiment.

The tests assume that there is an active connection to the MinIO bucket, because the experiment
configs use corpora from the "Scripture" and "Paratext" directories. The experiments themselves
are stored in the repository, to avoid having them accidentally changed or deleted, so every test
deletes the output of the steps that it runs before and after running them.
"""

import shutil
from pathlib import Path
from typing import Iterable, List

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.config import Config, NMTModel
from silnlp.nmt.experiment import SILExperiment
from silnlp.nmt.seq2seq_config import PreTrainedModelProviderFactory, Seq2SeqConfig

TEST_MT_DIR = Path(__file__).parent

# What each step writes to an experiment directory, as glob patterns. Everything else in an
# experiment directory is stored in the repository: the configs, the input files of the translate
# step, the test data of the test step and the checkpoint directory that both of them resolve.
PREPROCESS_OUTPUT_PATTERNS = [
    "train*",
    "val*",
    "test*",
    "dict*",
    "tokenizer*",
    "special_tokens*",
    "sentencepiece*",
    "tokenization_stats*",
    "token_occurrence*",
]
TRAIN_OUTPUT_PATTERNS = ["effective-config*", "run"]
TEST_OUTPUT_PATTERNS = ["test.trg-predictions*", "scores-*", "linregress*"]
TRANSLATE_OUTPUT_PATTERNS = ["infer"]

# What a run of the whole pipeline writes to an experiment directory
PIPELINE_OUTPUT_PATTERNS = (
    PREPROCESS_OUTPUT_PATTERNS + TRAIN_OUTPUT_PATTERNS + TEST_OUTPUT_PATTERNS + TRANSLATE_OUTPUT_PATTERNS
)


def set_up_environment() -> SilNlpEnv:
    return SilNlpEnv.create_environment_with_mt_experiments_dir(TEST_MT_DIR / "experiments")


def delete_generated_paths(experiment_directory: Path, patterns: Iterable[str]) -> None:
    """Delete the files and directories of an experiment directory that match the patterns."""
    for pattern in patterns:
        for path in experiment_directory.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def create_model_with_mock_pretrained_model(
    config: Config, model_provider_factory: PreTrainedModelProviderFactory
) -> NMTModel:
    """Create the model that a command would create, but with a mock pretrained model inside it.

    The commands don't take a pretrained model provider factory, so they create a model that
    downloads and runs the real pretrained model. Mixed precision is disabled, as the
    --disable-mixed-precision option does, because the tests run on the CPU.

    The config has to be a Seq2SeqConfig, because neither the factory nor the mixed precision
    parameter is part of the create_model signature that all configs have in common.
    """
    assert isinstance(config, Seq2SeqConfig)
    return config.create_model(mixed_precision=False, pretrained_model_provider_factory=model_provider_factory)


def create_full_pipeline_experiment(
    experiment_name: str, config: Config, model: NMTModel, environment: SilNlpEnv
) -> SILExperiment:
    """Create an experiment that runs every step, i.e. what silnlp.nmt.experiment.main creates."""
    return SILExperiment(
        name=experiment_name,
        config=config,
        model=model,
        environment=environment,
        run_prep=True,
        run_train=True,
        run_test=True,
        run_translate=True,
    )


def read_lines(path: Path) -> List[str]:
    """Read the lines of a generated file, without their line endings."""
    with path.open("r", encoding="utf-8-sig") as file:
        return [line.rstrip("\n") for line in file]


def count_lines(path: Path) -> int:
    return len(read_lines(path))
