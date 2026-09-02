from typing import Tuple

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.config_utils import load_config
from silnlp.nmt.experiment import SILExperiment
from silnlp.nmt.remote_llm_config import RemoteLLMConfig
from tests.smoke_tests.mock_completion_client import CompletionStats, MockCompletionClientFactory
from tests.smoke_tests.smoke_test_utils import (
    PIPELINE_OUTPUT_PATTERNS,
    count_lines,
    create_full_pipeline_experiment,
    delete_generated_paths,
    set_up_environment,
)

EXPERIMENT_NAME = "test_experiment_remote_llm"


def test_remote_llm_experiment_full_pipeline():
    # Like test_experiment.py, this runs the full pipeline and needs an active MinIO connection
    # for the "Scripture"/"Paratext" data. No LLM provider is contacted; the client is mocked.
    environment = set_up_environment()
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    delete_generated_paths(exp_dir, PIPELINE_OUTPUT_PATTERNS)

    experiment, stats = create_experiment_with_mock_client(environment)
    experiment.run()

    check_training_step(environment)
    check_test_step(environment)
    check_translate_step(environment)
    assert stats.num_requests > 0

    delete_generated_paths(exp_dir, PIPELINE_OUTPUT_PATTERNS)


def create_experiment_with_mock_client(environment: SilNlpEnv) -> Tuple[SILExperiment, CompletionStats]:
    factory = MockCompletionClientFactory()

    config = load_config(EXPERIMENT_NAME, environment)
    assert isinstance(config, RemoteLLMConfig)

    # A remote LLM model takes a completion client factory rather than a pretrained model provider, so
    # it cannot be created with create_model_with_mock_pretrained_model.
    model = config.create_model(completion_client_factory=factory)

    experiment = create_full_pipeline_experiment(EXPERIMENT_NAME, config, model, environment)
    return experiment, factory.stats


def check_training_step(environment: SilNlpEnv):
    # There is no fine-tuning; the train step builds the retrieval index and writes the
    # checkpoint that the test and translate steps resolve.
    checkpoint_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME) / "run" / "checkpoint-1"
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "retrieval.pkl").is_file()


def check_test_step(environment: SilNlpEnv):
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    predictions_path = exp_dir / "test.trg-predictions.detok.txt.1"
    assert predictions_path.exists()

    # There should be exactly one prediction line per test source sentence.
    assert count_lines(predictions_path) == count_lines(exp_dir / "test.src.txt")


def check_translate_step(environment: SilNlpEnv):
    infer_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME) / "infer"
    translated_files = list(infer_dir.glob("*/BSB/653JN.SFM"))
    assert len(translated_files) == 1
