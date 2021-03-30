import os
import random
import subprocess
from abc import ABC, abstractmethod
from enum import Flag
from pathlib import Path
from typing import Any, List

import numpy as np

from ..common.environment import MT_EXPERIMENTS_DIR


def get_repo_dir() -> Path:
    script_path = Path(__file__)
    return script_path.parent.parent.parent


def get_git_revision_hash() -> str:
    repo_dir = get_repo_dir()
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=10", "HEAD"], encoding="utf-8"
    ).strip()


def get_mt_exp_dir(exp_name: str) -> Path:
    return MT_EXPERIMENTS_DIR / exp_name


def set_seed(seed: Any) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


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


class NoiseMethod(ABC):
    @abstractmethod
    def __call__(self, tokens: List[str]) -> List[str]:
        pass


def random_bool(probability: float) -> bool:
    """Returns True with given probability

    Args:
        probability: probability to return True

    """
    assert 0 <= probability <= 1, "probability needs to be >= 0 and <= 1"
    return random.random() < probability


class DeleteRandomToken(NoiseMethod):
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if not random_bool(self.probability)]


class ReplaceRandomToken(NoiseMethod):
    def __init__(self, probability: float, filler_token: str = "<blank>") -> None:
        self.probability = probability
        self.filler_token = filler_token

    def __call__(self, tokens: List[str]) -> List[str]:
        new_tokens = tokens.copy()
        for i in range(len(new_tokens)):
            if random_bool(self.probability):
                new_tokens[i] = self.filler_token
        return new_tokens


class RandomTokenPermutation(NoiseMethod):
    def __init__(self, distance: int) -> None:
        self.distance = distance

    def __call__(self, tokens: List[str]) -> List[str]:
        new_indices = [i + random.uniform(0, self.distance + 1) for i in range(len(tokens))]
        return [x for _, x in sorted(zip(new_indices, tokens), key=lambda pair: pair[0])]
