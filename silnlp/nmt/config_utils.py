import shutil
from pathlib import Path

import yaml

from ..common.environment import SilNlpEnv
from .config import Config
from .model_name import ModelName
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


class ConfiguredModelType:
    """Which kind of model a config targets. An explicit ``model_type: llm`` wins; otherwise the model
    name decides. Detection is string-only by design - we never load the model's AutoConfig here,
    since create_config is on the hot path of every CLI command."""

    def __init__(self, config: dict) -> None:
        self._config = config

    def is_llm(self) -> bool:
        model_type = self._config.get("model_type")
        if model_type is not None:
            return str(model_type).lower() == "llm"
        return ModelName(self._config.get("model", "")).looks_decoder_only()


def create_config(exp_dir: Path, config: dict, environment: SilNlpEnv) -> Config:
    if ConfiguredModelType(config).is_llm():
        # Imported lazily so the peft/bitsandbytes import cost is only paid for LLM experiments.
        from .llm_config import LLMConfig

        return LLMConfig(exp_dir, config, environment)
    return Seq2SeqConfig(exp_dir, config, environment)
