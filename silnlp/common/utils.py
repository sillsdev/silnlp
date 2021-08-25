import logging
import os
import random
import subprocess
from abc import ABC, abstractmethod
from enum import Flag
from pathlib import Path
from typing import Any, List, Optional, Set

import numpy as np

from ..common.environment import SIL_NLP_ENV

LOGGER = logging.getLogger(__name__)


def get_repo_dir() -> Path:
    script_path = Path(__file__)
    return script_path.parent.parent.parent


def get_git_revision_hash() -> str:
    repo_dir = get_repo_dir()
    git_hash = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=10", "HEAD"], encoding="utf-8"
    ).strip()
    LOGGER.info("Git commit: " + git_hash)
    return git_hash


def get_mt_exp_dir(exp_name: str) -> Path:
    return SIL_NLP_ENV.mt_experiments_dir / exp_name


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


def unique_list(seq: List[str]) -> List[str]:
    # make the lists unique, keeping only the first element found
    seen: Set[str] = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def is_set(value: Flag, flag: Flag) -> bool:
    return (value & flag) == flag


_is_dotnet_installed: Optional[bool] = None


def check_dotnet() -> None:
    global _is_dotnet_installed
    if _is_dotnet_installed is None:
        # Update or add dotnet machine environment
        try:
            subprocess.run(
                ["dotnet", "tool", "restore"],
                cwd=Path(__file__).parent.parent.parent,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _is_dotnet_installed = True
        except:
            _is_dotnet_installed = False

    if not _is_dotnet_installed:
        raise RuntimeError("The .NET Core SDK needs to be installed (https://dotnet.microsoft.com/download).")


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
