from pathlib import Path
from typing import Optional

import yaml

from ..common.utils import get_mt_exp_dir
from .config import Config
from .hugging_face_config import HuggingFaceConfig


def load_config(exp_name: str, use_default_model_dir: bool = True) -> Config:
    exp_dir = get_mt_exp_dir(exp_name)
    config_path = exp_dir / "config.yml"

    with config_path.open("r", encoding="utf-8") as file:
        config: dict = yaml.safe_load(file)
    return create_config(exp_dir, config, use_default_model_dir)


def create_config(exp_dir: Path, config: dict, use_default_model_dir: bool = True) -> Config:
    return HuggingFaceConfig(exp_dir, config, use_default_model_dir)
