from pathlib import Path

import yaml

from .utils import get_mt_exp_dir
from ..nmt.config import Config
from ..nmt.hugging_face_config import HuggingFaceConfig as NMTHuggingFaceConfig
from ..asr.hugging_face_config import HuggingFaceConfig as ASRHuggingFaceConfig


def load_config(exp_name: str) -> Config:
    exp_dir = get_mt_exp_dir(exp_name)
    config_path = exp_dir / "config.yml"

    with config_path.open("r", encoding="utf-8") as file:
        config: dict = yaml.safe_load(file)
    return create_nmt_config(exp_dir, config)


def create_nmt_config(exp_dir: Path, config: dict) -> Config:
    return NMTHuggingFaceConfig(exp_dir, config)

def create_asr_config(exp_dir: Path, config: dict) -> Config:
    return ASRHuggingFaceConfig(exp_dir, config)
