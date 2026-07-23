import shutil
from pathlib import Path
from typing import cast

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.config_utils import load_config_from_exp_dir
from silnlp.nmt.experiment import SILExperiment
from silnlp.nmt.llm_config import LLMConfig
from tests.smoke_tests.mock_causal_model import CausalModelTrainingStats, MockCausalLMProviderFactory

TEST_MT_DIR = Path(__file__).parent
EXPERIMENT_NAME = "test_experiment_llm"


def test_llm_experiment_full_pipeline():
    # Like test_experiment.py, this exercises the full pipeline (preprocess -> train -> test ->
    # translate) and assumes an active MinIO connection for the "Scripture"/"Paratext" data.
    environment = set_up_environment()
    clean_experiment_directory(environment.get_mt_exp_dir(EXPERIMENT_NAME))

    experiment, model_stats = create_experiment_with_mock_model(environment)
    experiment.run()

    check_training_step(model_stats)
    check_test_step(environment)
    check_translate_step(environment)

    clean_experiment_directory(environment.get_mt_exp_dir(EXPERIMENT_NAME))


def set_up_environment() -> SilNlpEnv:
    return SilNlpEnv.create_environment_with_mt_experiments_dir(TEST_MT_DIR / "experiments")


def clean_experiment_directory(experiment_directory: Path):
    for pattern in ("train*", "test*", "val*", "scores*", "effective-config*"):
        for file in experiment_directory.glob(pattern):
            file.unlink()
    for sub in ("run", "infer"):
        sub_dir = experiment_directory / sub
        if sub_dir.is_dir():
            shutil.rmtree(sub_dir)


def create_experiment_with_mock_model(environment: SilNlpEnv) -> tuple[SILExperiment, CausalModelTrainingStats]:
    factory = MockCausalLMProviderFactory()

    config = cast(LLMConfig, load_config_from_exp_dir(environment.get_mt_exp_dir(EXPERIMENT_NAME), environment))
    assert isinstance(config, LLMConfig)

    model = config.create_model(pretrained_model_provider_factory=factory)
    experiment = SILExperiment(
        name=EXPERIMENT_NAME,
        config=config,
        model=model,
        environment=environment,
        run_prep=True,
        run_train=True,
        run_test=True,
        run_translate=True,
    )
    return experiment, factory.stats


def check_training_step(model_stats: CausalModelTrainingStats):
    assert model_stats.num_forward_calls > 0


def check_test_step(environment: SilNlpEnv):
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    predictions_path = exp_dir / "test.trg-predictions.detok.txt.8"
    assert predictions_path.exists()

    # There should be exactly one prediction line per test source sentence.
    num_sources = sum(1 for _ in (exp_dir / "test.src.txt").open("r", encoding="utf-8-sig"))
    num_predictions = sum(1 for _ in predictions_path.open("r", encoding="utf-8"))
    assert num_predictions == num_sources


def check_translate_step(environment: SilNlpEnv):
    infer_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME) / "infer"
    translated_files = list(infer_dir.glob("*/BSB/653JN.SFM"))
    assert len(translated_files) == 1
