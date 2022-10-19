from pathlib import Path
from typing import Optional

import yaml

from ..common.utils import get_mt_exp_dir
from .config import Config
from .hugging_face_config import HuggingFaceConfig
from .open_nmt_config import OpenNMTConfig, is_open_nmt_model


def load_config(exp_name: str) -> Config:
    exp_dir = get_mt_exp_dir(exp_name)
    config_path = exp_dir / "config.yml"

    with config_path.open("r", encoding="utf-8") as file:
        config: dict = yaml.safe_load(file)
    return create_config(exp_dir, config)


def create_config(exp_dir: Path, config: dict) -> Config:
    model_name: Optional[str] = config.get("model")
    if model_name is None or is_open_nmt_model(model_name):
        return OpenNMTConfig(exp_dir, config)

    return HuggingFaceConfig(exp_dir, config)
