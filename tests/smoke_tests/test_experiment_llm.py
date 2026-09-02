from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.config_utils import load_config
from silnlp.nmt.experiment import SILExperiment
from silnlp.nmt.llm_config import LLMConfig
from tests.smoke_tests.mock_causal_model import CausalModelTrainingStats, MockCausalLMProviderFactory
from tests.smoke_tests.smoke_test_utils import (
    PIPELINE_OUTPUT_PATTERNS,
    count_lines,
    create_full_pipeline_experiment,
    delete_generated_paths,
    set_up_environment,
)

EXPERIMENT_NAME = "test_experiment_llm"


def test_llm_experiment_full_pipeline():
    # Like test_experiment.py, this exercises the full pipeline (preprocess -> train -> test ->
    # translate) and assumes an active MinIO connection for the "Scripture"/"Paratext" data.
    environment = set_up_environment()
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    delete_generated_paths(exp_dir, PIPELINE_OUTPUT_PATTERNS)

    experiment, model_stats = create_experiment_with_mock_model(environment)
    experiment.run()

    check_training_step(model_stats)
    check_test_step(environment)
    check_translate_step(environment)

    delete_generated_paths(exp_dir, PIPELINE_OUTPUT_PATTERNS)


def create_experiment_with_mock_model(environment: SilNlpEnv) -> tuple[SILExperiment, CausalModelTrainingStats]:
    factory = MockCausalLMProviderFactory()

    config = load_config(EXPERIMENT_NAME, environment)
    assert isinstance(config, LLMConfig)

    # A decoder-only model takes a different kind of provider factory than a seq2seq model, so the
    # model cannot be created with create_model_with_mock_pretrained_model
    model = config.create_model(pretrained_model_provider_factory=factory)

    experiment = create_full_pipeline_experiment(EXPERIMENT_NAME, config, model, environment)
    return experiment, factory.stats


def check_training_step(model_stats: CausalModelTrainingStats):
    assert model_stats.num_forward_calls > 0


def check_test_step(environment: SilNlpEnv):
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    predictions_path = exp_dir / "test.trg-predictions.detok.txt.8"
    assert predictions_path.exists()

    # There should be exactly one prediction line per test source sentence.
    assert count_lines(predictions_path) == count_lines(exp_dir / "test.src.txt")


def check_translate_step(environment: SilNlpEnv):
    infer_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME) / "infer"
    translated_files = list(infer_dir.glob("*/BSB/653JN.SFM"))
    assert len(translated_files) == 1
