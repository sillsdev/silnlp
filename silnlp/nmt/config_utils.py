from pathlib import Path

import yaml

from ..common.environment import SilNlpEnv
from .config import Config
from .hugging_face_config import HuggingFaceConfig


def load_config(exp_name: str, environment: SilNlpEnv) -> Config:
    exp_dir = environment.get_mt_exp_dir(exp_name)
    config_path = exp_dir / "config.yml"

    with config_path.open("r", encoding="utf-8") as file:
        config: dict = yaml.safe_load(file)
    return create_config(exp_dir, config, environment)


def load_config_from_exp_dir(exp_dir: Path, environment: SilNlpEnv) -> Config:
    config_path = exp_dir / "config.yml"

    with config_path.open("r", encoding="utf-8") as file:
        config: dict = yaml.safe_load(file)
    return create_config(exp_dir, config, environment)


def create_config(exp_dir: Path, config: dict, environment: SilNlpEnv) -> Config:
    return HuggingFaceConfig(exp_dir, config, environment)
