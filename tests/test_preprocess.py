import pytest
from pathlib import Path
import shutil
import os
from . import helper
from silnlp.nmt.config import load_config


exp_truth_dir = Path(__file__).parent / "MT/Experiment_Truth"

exp_dir = Path(__file__).parent / "MT/Experiments"
exp_subdirs = [folder for folder in exp_truth_dir.glob("*/")]


@pytest.mark.parametrize("exp_folder", exp_subdirs)
def test_preprocess(exp_folder):
    exp_truth_path = os.path.join(exp_truth_dir, exp_folder)
    config_file = os.path.join(exp_truth_path, "config.yml")
    assert os.path.isfile(config_file), "The configuraiton file config.yml does not exist for " + exp_folder.name
    experiment_path = exp_dir / exp_folder.name
    shutil.rmtree(experiment_path, ignore_errors=True)
    os.mkdir(experiment_path)
    shutil.copyfile(src=config_file, dst=os.path.join(experiment_path, "config.yml"))

    config = load_config(experiment_path)
    config.set_seed()
    config.preprocess(stats=False)

    helper.compare_folders(truth_folder=exp_truth_path, computed_folder=experiment_path)
