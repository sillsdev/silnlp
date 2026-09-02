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


# Decoder-only LLM model name prefixes used as a fallback when "model_type" is not set.
LLM_MODEL_PREFIXES = (
    "google/gemma",
    "google/translate-gemma",
    "google/translategemma",
    "tencent/Hunyuan",
    "Hunyuan-MT",
)


def is_llm_config(config: dict) -> bool:
    """Decide whether a config targets a decoder-only LLM.

    An explicit ``model_type: llm`` wins; otherwise fall back to a string prefix match on the
    model name. Detection is string-only by design - we never load the model's AutoConfig here,
    since create_config is on the hot path of every CLI command.
    """
    model_type = config.get("model_type")
    if model_type is not None:
        return str(model_type).lower() == "llm"
    model: str = config.get("model", "")
    return any(model.startswith(prefix) for prefix in LLM_MODEL_PREFIXES)


def create_config(exp_dir: Path, config: dict, environment: SilNlpEnv) -> Config:
    if is_llm_config(config):
        # Imported lazily so the peft/bitsandbytes import cost is only paid for LLM experiments.
        from .llm_config import LLMConfig

        return LLMConfig(exp_dir, config, environment)
    return Seq2SeqConfig(exp_dir, config, environment)
