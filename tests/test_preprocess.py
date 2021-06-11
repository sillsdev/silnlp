import pytest
from pathlib import Path
import shutil
import os

exp_truth_dir = Path(__file__).parent / "MT/Experiment_Truth"
# subdirs = [folder for folder in exp_dir.glob("*/")]
subdirs = ["ABP-ABPBTE", "AMIU-AMIBT"]

exp_dir = Path(__file__).parent / "MT/Experiments"


@pytest.mark.parametrize("experiment_folder", subdirs)
def test_preprocess(experiment_folder):
    config_file = os.path.join(exp_truth_dir,experiment_folder,"config.yml")
    assert os.path.isfile(config_file), 
        'The configuraiton file config.yml does not exist for ' + experiment_folder
#    shutil.copyfile(src=config_file,
#    dst=os.path.join(exp_truth_dir,experiment_folder,"config.yml"))
    print(experiment_path)
    assert experiment_path is not None


def test_other():
    assert False