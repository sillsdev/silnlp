import shutil
from pathlib import Path

import yaml

from ..common.environment import SilNlpEnv
from .config import Config
from .seq2seq_config import Seq2SeqConfig


class ExperimentConfigFile:
    """The config.yml an experiment runs from. An experiment with none of its own inherits the one
    its project folder holds, which is copied down so the experiment keeps a record of what it ran."""

    _NAME = "config.yml"

    def __init__(self, exp_dir: Path, project_dir: Path) -> None:
        self._exp_dir = exp_dir
        self._project_dir = project_dir

    def read(self) -> dict:
        self._inherit_project_config()
        path = self._exp_dir / self._NAME
        if not path.exists():
            raise RuntimeError(f"No config file at {path}.")
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if config is None or len(config.keys()) == 0:
            raise RuntimeError("Config file has no contents.")
        return config

    def _inherit_project_config(self) -> None:
        if (self._exp_dir / self._NAME).exists():
            return
        project_config = self._project_dir / self._NAME
        if not project_config.exists():
            return
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(project_config), str(self._exp_dir / self._NAME))


def experiment_config_file(exp_name: str, environment: SilNlpEnv) -> ExperimentConfigFile:
    return ExperimentConfigFile(
        environment.get_mt_exp_dir(exp_name), environment.get_mt_exp_dir(exp_name.split("/")[0])
    )


def load_config(exp_name: str, environment: SilNlpEnv) -> Config:
    exp_dir = environment.get_mt_exp_dir(exp_name)
    config = experiment_config_file(exp_name, environment).read()
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
