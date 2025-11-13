import logging
import os
import random
import subprocess
from abc import ABC, abstractmethod
from enum import Enum, Flag, auto
from inspect import getmembers
from pathlib import Path, PurePath
from types import FunctionType
from typing import Any, List, Optional, Set, Type, cast

import numpy as np
import pandas as pd

from ..common.environment import SIL_NLP_ENV

LOGGER = logging.getLogger(__name__)


class Side(Enum):
    SOURCE = auto()
    TARGET = auto()


def api(obj):
    return [name for name in dir(obj) if name[0] != "_"]


def attrs(obj):
    disallowed_properties = {
        name for name, value in getmembers(type(obj)) if isinstance(value, (property, FunctionType))
    }
    return {name: getattr(obj, name) for name in api(obj) if name not in disallowed_properties and hasattr(obj, name)}


def print_table(rows):

    for arg, value in rows:
        if isinstance(value, PurePath):
            print(f"{str(arg):<30} : {str(value):<41} | {str(type(value)):<30} | {str(value.exists()):>6}")
        else:
            print(f"{str(arg):<30} : {str(value):<41} | {str(type(value)):<30}")

    print()


def show_attrs(cli_args, envs=SIL_NLP_ENV, actions=[]):

    env_rows = [(k, v) for k, v in attrs(envs).items()]
    arg_rows = [(k, v) for k, v in cli_args.__dict__.items() if v is not None]

    print("\nEnvironment Variables:")
    print_table(env_rows)

    print("Command line arguments:")
    print_table(arg_rows)

    for action in actions:
        print(action)


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
    def __call__(self, tokens: list) -> list:
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

    def __call__(self, tokens: list) -> list:
        return [token for token in tokens if not random_bool(self.probability)]


class ReplaceRandomToken(NoiseMethod):
    def __init__(self, probability: float, filler_token: str = "<blank>") -> None:
        self.probability = probability
        self.filler_token = filler_token

    def __call__(self, tokens: list) -> list:
        new_tokens = tokens.copy()
        for i in range(len(new_tokens)):
            if random_bool(self.probability):
                new_tokens[i] = self.filler_token
        return new_tokens


class RandomTokenPermutation(NoiseMethod):
    def __init__(self, distance: int) -> None:
        self.distance = distance

    def __call__(self, tokens: list) -> list:
        new_indices = [i + random.uniform(0, self.distance + 1) for i in range(len(tokens))]
        return [x for _, x in sorted(zip(new_indices, tokens), key=lambda pair: pair[0])]


def create_noise_methods(params: List[dict]) -> List[NoiseMethod]:
    methods: List[NoiseMethod] = []
    for module in params:
        noise_type, args = next(iter(module.items()))
        if not isinstance(args, list):
            args = [args]
        noise_type = noise_type.lower()
        noise_method_class: Type[NoiseMethod]
        if noise_type == "dropout":
            noise_method_class = DeleteRandomToken
        elif noise_type == "replacement":
            noise_method_class = ReplaceRandomToken
        elif noise_type == "permutation":
            noise_method_class = RandomTokenPermutation
        else:
            raise ValueError("Invalid noise type: %s" % noise_type)
        methods.append(noise_method_class(*args))
    return methods


def _get_tags_str(tags: Optional[List[str]]) -> str:
    tags_str = ""
    if tags is not None and len(tags) > 0:
        tags_str += " ".join(f"<{t}>" for t in tags) + " "
    return tags_str


def add_tags_to_sentence(tags: Optional[List[str]], sentence: str) -> str:
    return _get_tags_str(tags) + sentence


def add_tags_to_dataframe(tags: Optional[List[str]], df_sentences: pd.DataFrame) -> pd.DataFrame:
    tags_str = _get_tags_str(tags)
    if tags_str != "":
        cast(Any, df_sentences).loc[:, "source"] = tags_str + df_sentences.loc[:, "source"]
    return df_sentences
