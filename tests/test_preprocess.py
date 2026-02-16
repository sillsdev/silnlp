import os
import shutil

import pytest
import sentencepiece as sp

from silnlp.common.environment import SIL_NLP_ENV
from silnlp.common.config_utils import load_config

from . import helper

# set experiment directory to temp
SIL_NLP_ENV.set_data_dir()
SIL_NLP_ENV.mt_experiments_dir = SIL_NLP_ENV.mt_dir / "temp_experiments"
SIL_NLP_ENV.mt_experiments_dir.mkdir(exist_ok=True)
exp_truth_dir = SIL_NLP_ENV.mt_dir / "Experiments"
exp_subdirs = [folder for folder in exp_truth_dir.glob("*/")]


@pytest.mark.parametrize("exp_folder", exp_subdirs)
def test_preprocess(exp_folder):
    exp_truth_path = os.path.join(exp_truth_dir, exp_folder)
    config_file = os.path.join(exp_truth_path, "config.yml")
    assert os.path.isfile(config_file), "The configuration file config.yml does not exist for " + exp_folder.name
    experiment_path = SIL_NLP_ENV.mt_experiments_dir / exp_folder.name
    shutil.rmtree(experiment_path, ignore_errors=True)
    os.makedirs(experiment_path, exist_ok=True)
    shutil.copyfile(src=config_file, dst=os.path.join(experiment_path, "config.yml"))
    helper.init_file_logger(str(experiment_path))

    sp.set_random_generator_seed(111)  # this is to make the vocab generation consistent
    config = load_config(experiment_path)
    config.set_seed()
    config.preprocess(stats=False)

    helper.compare_folders(truth_folder=exp_truth_path, computed_folder=experiment_path)
