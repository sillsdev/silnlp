import os
import sys
import random
import subprocess
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

from ..common.environment import ALIGN_EXPERIMENTS_DIR, PT_PREPROCESSED_DIR


def get_repo_dir() -> str:
    script_path = Path(__file__)
    return script_path.parent.parent.parent


def get_git_revision_hash() -> str:
    repo_dir = get_repo_dir()
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=10", "HEAD"], encoding="utf-8"
    ).strip()


def get_mt_root_dir(exp_name: str) -> str:
    mt_root_dir = os.path.join(PT_PREPROCESSED_DIR, "tests", exp_name)
    mt_root_path = Path(mt_root_dir)

    if not mt_root_path.exists():
        sys.exit(f"\nExperiement folder missing: {mt_root_path}\n")	

    return mt_root_dir


def get_align_root_dir(exp_name: str) -> str:
    alignments_dir = os.path.join(ALIGN_EXPERIMENTS_DIR, exp_name)
    alignments_path = Path(alignments_dir)
    
    if not alignments_path.exists():
        sys.exit(f"\nAlignments folder missing:\n{alignments_path}\n")

    return alignments_dir


def set_seed(seed: Any) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def wsl_path(win_path: str) -> str:
    win_path = os.path.normpath(win_path).replace("\\", "\\\\")
    if sys.version_info < (3, 7, 0):
        result = subprocess.run(
            ["wsl", "wslpath", "-a", win_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
        )
    else:
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
