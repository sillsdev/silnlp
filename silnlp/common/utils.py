import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from nlp.common.environment import paratextPreprocessedDir


def get_git_revision_hash() -> str:
    script_path = Path(__file__)
    repo_dir = script_path.parent.parent.parent
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=10", "HEAD"], encoding="utf-8"
    ).strip()


def get_root_dir(exp_name: str) -> str:
    return os.path.join(paratextPreprocessedDir, "tests", exp_name)


def set_seed(seed: Any) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
