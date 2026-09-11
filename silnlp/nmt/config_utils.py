from pathlib import Path

import yaml

from ..common.environment import SilNlpEnv
from .config import Config
from .seq2seq_config import Seq2SeqConfig


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


def is_local_llm_config(config: dict) -> bool:
    model_type = config.get("model_type")
    if model_type is not None:
        # "local_llm" is the preferred label, but "llm" is supported for backward compatibility
        return str(model_type).lower() in ("local_llm", "llm")
    # Matching the model name is a fallback when "model_type" is not set.
    model: str = config.get("model", "")
    return model.startswith(
        ("google/gemma", "google/translate-gemma", "google/translategemma", "tencent/Hunyuan", "Hunyuan-MT")
    )


def is_remote_llm_config(config: dict) -> bool:
    return str(config.get("model_type", "")).lower() == "remote_llm"


def create_config(exp_dir: Path, config: dict, environment: SilNlpEnv) -> Config:
    if is_remote_llm_config(config):
        # Imported lazily so the litellm import cost is only paid for remote LLM experiments.
        from .remote_llm_config import RemoteLLMConfig

        return RemoteLLMConfig(exp_dir, config, environment)
    if is_local_llm_config(config):
        # Imported lazily so the peft/bitsandbytes import cost is only paid for LLM experiments.
        from .local_llm_config import LocalLLMConfig

        return LocalLLMConfig(exp_dir, config, environment)
    return Seq2SeqConfig(exp_dir, config, environment)
