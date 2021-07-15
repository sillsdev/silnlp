import pytest
from pathlib import Path
import shutil
import tempfile
import os
from . import helper
from silnlp.nmt.config import load_config
from silnlp.common.environment import SNE


exp_truth_dir = SNE._MT_DIR / "Experiments"
exp_subdirs = [folder for folder in exp_truth_dir.glob("*/")]

# set experiment directory to temp
with tempfile.TemporaryDirectory() as src_dir:
    SNE._MT_EXPERIMENTS_DIR = Path(src_dir)


@pytest.mark.parametrize("exp_folder", exp_subdirs)
def test_preprocess(exp_folder):
    exp_truth_path = os.path.join(exp_truth_dir, exp_folder)
    config_file = os.path.join(exp_truth_path, "config.yml")
    assert os.path.isfile(config_file), "The configuraiton file config.yml does not exist for " + exp_folder.name
    experiment_path = SNE._MT_EXPERIMENTS_DIR / exp_folder.name
    shutil.rmtree(experiment_path, ignore_errors=True)
    os.makedirs(experiment_path, exist_ok=True)
    shutil.copyfile(src=config_file, dst=os.path.join(experiment_path, "config.yml"))
    helper.init_file_logger(str(experiment_path))

    config = load_config(experiment_path)
    config.set_seed()
    config.preprocess(stats=False)

    helper.compare_folders(truth_folder=exp_truth_path, computed_folder=experiment_path)
