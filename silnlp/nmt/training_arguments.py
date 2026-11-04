from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml


class TrainingArgumentsMapping:
    """Which experiment config values become HuggingFace TrainingArguments fields. The seq2seq and LLM
    models differ only in this mapping, the arguments class, and the precision flags they pass."""

    def __init__(self, mapping: Dict[str, Set[str]], config_root: dict) -> None:
        self._mapping = mapping
        self._config_root = config_root

    def collect(self, precision_args: Dict[str, Any], clearml_queue: Optional[str]) -> Dict[str, Any]:
        config_root = self._config_root
        args: Dict[str, Any] = {}
        for section, params in self._mapping.items():
            section_config: dict = config_root[section]
            for param in params:
                if param in section_config and section_config[param] is not None:
                    args[param] = section_config[param]
        args.update(precision_args)
        args["report_to"] = "none" if clearml_queue is None else "all"
        return args

    def write_effective_config(self, path: Path, training_args: Any) -> None:
        """Write the resolved experiment config, overlaying onto a copy of it the values the training
        arguments actually settled on."""
        config = deepcopy(self._config_root)
        for section, params in self._mapping.items():
            section_config: dict = config[section]
            for param in params:
                value = getattr(training_args, param)
                if isinstance(value, Enum):
                    value = value.value
                if value is None:
                    section_config.pop(param, None)
                else:
                    section_config[param] = value
        with path.open("w") as file:
            yaml.dump(config, file)
