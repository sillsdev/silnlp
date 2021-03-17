import os
import random
import subprocess
from enum import Flag
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from ..common.environment import PT_PREPROCESSED_DIR


def get_repo_dir() -> str:
    script_path = Path(__file__)
    return str(script_path.parent.parent.parent)


def get_git_revision_hash() -> str:
    repo_dir = get_repo_dir()
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=10", "HEAD"], encoding="utf-8"
    ).strip()


def get_mt_exp_dir(exp_name: str) -> str:
    return os.path.join(PT_PREPROCESSED_DIR, "tests", exp_name)


def set_seed(seed: Any) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


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


def is_set(value: Flag, flag: Flag) -> bool:
    return (value & flag) == flag
