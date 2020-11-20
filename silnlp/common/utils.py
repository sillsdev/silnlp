import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from ..common.environment import ALIGN_EXPERIMENTS_DIR, PT_PREPROCESSED_DIR


def get_git_revision_hash() -> str:
    script_path = Path(__file__)
    repo_dir = script_path.parent.parent.parent
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=10", "HEAD"], encoding="utf-8"
    ).strip()


def get_mt_root_dir(exp_name: str) -> str:
    return os.path.join(PT_PREPROCESSED_DIR, "tests", exp_name)


def get_align_root_dir(exp_name: str) -> str:
    return os.path.join(ALIGN_EXPERIMENTS_DIR, exp_name)


def set_seed(seed: Any) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def wsl_path(win_path: str) -> str:
    win_path = os.path.normpath(win_path).replace("\\", "\\\\")
    result = subprocess.run(["wsl", "wslpath", "-a", win_path], capture_output=True, encoding="utf-8")
    return result.stdout.strip()


def merge_dict(dict1: dict, dict2: dict) -> dict:
    for key, value in dict2.items():
        if isinstance(value, dict):
            dict1_value = dict1.get(key, {})
            if isinstance(dict1_value, dict):
                dict1[key] = merge_dict(dict1_value, value)
            else:
                dict1[key] = value
        else:
            dict1[key] = value
    return dict1
