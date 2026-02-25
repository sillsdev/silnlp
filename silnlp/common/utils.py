import logging
import os
import random
import subprocess
from abc import ABC, abstractmethod
from argparse import Namespace
from enum import Enum, Flag, auto
from inspect import getmembers
from pathlib import Path, PurePath
from types import FunctionType
from typing import Any, List, Optional, Set, Type, cast

import numpy as np
import pandas as pd

from ..common.environment import SIL_NLP_ENV, SilNlpEnv

LOGGER = logging.getLogger(__name__)

two2three_iso = {
    "aa": "aar",
    "ab": "abk",
    "af": "afr",
    "ak": "aka",
    "am": "amh",
    "ar": "ara",
    "an": "arg",
    "as": "asm",
    "av": "ava",
    "ae": "ave",
    "ay": "aym",
    "az": "aze",
    "ba": "bak",
    "bm": "bam",
    "be": "bel",
    "bn": "ben",
    "bi": "bis",
    "bo": "bod",
    "bs": "bos",
    "br": "bre",
    "bg": "bul",
    "ca": "cat",
    "cs": "ces",
    "ch": "cha",
    "ce": "che",
    "cu": "chu",
    "cv": "chv",
    "kw": "cor",
    "co": "cos",
    "cr": "cre",
    "cy": "cym",
    "da": "dan",
    "de": "deu",
    "dv": "div",
    "dz": "dzo",
    "el": "ell",
    "en": "eng",
    "eo": "epo",
    "et": "est",
    "eu": "eus",
    "ee": "ewe",
    "fo": "fao",
    "fa": "fas",
    "fj": "fij",
    "fi": "fin",
    "fr": "fra",
    "fy": "fry",
    "ff": "ful",
    "gd": "gla",
    "ga": "gle",
    "gl": "glg",
    "gv": "glv",
    "gn": "grn",
    "gu": "guj",
    "ht": "hat",
    "ha": "hau",
    "sh": "hbs",
    "he": "heb",
    "hz": "her",
    "hi": "hin",
    "ho": "hmo",
    "hr": "hrv",
    "hu": "hun",
    "hy": "hye",
    "ig": "ibo",
    "io": "ido",
    "ii": "iii",
    "iu": "iku",
    "ie": "ile",
    "ia": "ina",
    "id": "ind",
    "ik": "ipk",
    "is": "isl",
    "it": "ita",
    "jv": "jav",
    "ja": "jpn",
    "kl": "kal",
    "kn": "kan",
    "ks": "kas",
    "ka": "kat",
    "kr": "kau",
    "kk": "kaz",
    "km": "khm",
    "ki": "kik",
    "rw": "kin",
    "ky": "kir",
    "kv": "kom",
    "kg": "kon",
    "ko": "kor",
    "kj": "kua",
    "ku": "kur",
    "lo": "lao",
    "la": "lat",
    "lv": "lav",
    "li": "lim",
    "ln": "lin",
    "lt": "lit",
    "lb": "ltz",
    "lu": "lub",
    "lg": "lug",
    "mh": "mah",
    "ml": "mal",
    "mr": "mar",
    "mk": "mkd",
    "mg": "mlg",
    "mt": "mlt",
    "mn": "mon",
    "mi": "mri",
    "ms": "msa",
    "my": "mya",
    "na": "nau",
    "nv": "nav",
    "nr": "nbl",
    "nd": "nde",
    "ng": "ndo",
    "ne": "nep",
    "nl": "nld",
    "nn": "nno",
    "nb": "nob",
    "no": "nor",
    "ny": "nya",
    "oc": "oci",
    "oj": "oji",
    "or": "ori",
    "om": "orm",
    "os": "oss",
    "pa": "pan",
    "pi": "pli",
    "pl": "pol",
    "pt": "por",
    "ps": "pus",
    "qu": "que",
    "rm": "roh",
    "ro": "ron",
    "rn": "run",
    "ru": "rus",
    "sg": "sag",
    "sa": "san",
    "si": "sin",
    "sk": "slk",
    "sl": "slv",
    "se": "sme",
    "sm": "smo",
    "sn": "sna",
    "sd": "snd",
    "so": "som",
    "st": "sot",
    "es": "spa",
    "sq": "sqi",
    "sc": "srd",
    "sr": "srp",
    "ss": "ssw",
    "su": "sun",
    "sw": "swa",
    "sv": "swe",
    "ty": "tah",
    "ta": "tam",
    "tt": "tat",
    "te": "tel",
    "tg": "tgk",
    "tl": "tgl",
    "th": "tha",
    "ti": "tir",
    "to": "ton",
    "tn": "tsn",
    "ts": "tso",
    "tk": "tuk",
    "tr": "tur",
    "tw": "twi",
    "ug": "uig",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "ve": "ven",
    "vi": "vie",
    "vo": "vol",
    "wa": "wln",
    "wo": "wol",
    "xh": "xho",
    "yi": "yid",
    "yo": "yor",
    "za": "zha",
    "zh": "zho",
    "zu": "zul",
}


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


def show_attrs(cli_args: Namespace, envs: SilNlpEnv = SIL_NLP_ENV, actions: List[str] = []) -> None:

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
        except Exception:
            _is_dotnet_installed = False

    if not _is_dotnet_installed:
        raise RuntimeError("The .NET Core SDK needs to be installed (https://dotnet.microsoft.com/download).")


class NoiseMethod(ABC):
    @abstractmethod
    def __call__(self, tokens: list[str]) -> list[str]:
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
