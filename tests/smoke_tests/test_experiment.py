from pathlib import Path

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.config_utils import load_config_from_exp_dir
from silnlp.nmt.experiment import SILExperiment

TEST_MT_DIR = Path(__file__).parent
EXPERIMENT_NAME = "test_experiment"


def test_experiment_full_pipeline():
    environment = SilNlpEnv()
    environment.set_machine_translation_dir(TEST_MT_DIR)

    _clean_experiment_directory(environment.get_mt_exp_dir(EXPERIMENT_NAME))

    # Load the config from the test experiment directory
    config = load_config_from_exp_dir(environment.get_mt_exp_dir(EXPERIMENT_NAME))

    # Create the model from the config
    model = config.create_model()

    experiment = SILExperiment(
        name=EXPERIMENT_NAME,
        config=config,
        model=model,
        environment=environment,
        run_prep=True,
        run_train=False,
        run_test=False,
        run_translate=True,
    )
    experiment.run()


def _clean_experiment_directory(experiment_directory: Path):
    if experiment_directory.exists():
        for file in experiment_directory.glob("train*"):
            file.unlink()
        for file in experiment_directory.glob("test*"):
            file.unlink()
        for file in experiment_directory.glob("val*"):
            file.unlink()
        for file in experiment_directory.glob("tokenizer*"):
            file.unlink()
        for file in experiment_directory.glob("special_tokens*"):
            file.unlink()
        for file in experiment_directory.glob("sentencepiece*"):
            file.unlink()

        run_dir = experiment_directory / "run"
        if run_dir.exists() and run_dir.is_dir():
            for file in run_dir.glob("*"):
                file.unlink()
            run_dir.rmdir()

        infer_dir = experiment_directory / "infer"
        if infer_dir.exists() and infer_dir.is_dir():
            for file in infer_dir.glob("*"):
                file.unlink()
            infer_dir.rmdir()
