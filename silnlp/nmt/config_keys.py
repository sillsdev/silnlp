import logging
from typing import Dict

LOGGER = logging.getLogger(__name__)


class RenamedConfigKeys:
    """Config keys that HuggingFace renamed between 4.x and 5.x. A config still using an old name is
    warned about rather than having the setting silently dropped."""

    _COMMON = {
        "train": {"group_by_length": "train_sampling_strategy"},
        "params": {"warmup_ratio": "warmup_steps"},
    }

    @classmethod
    def of_seq2seq(cls) -> "RenamedConfigKeys":
        return cls({**cls._COMMON, "eval": {"include_inputs_for_metrics": "include_for_metrics"}})

    @classmethod
    def of_llm(cls) -> "RenamedConfigKeys":
        return cls(cls._COMMON)

    def __init__(self, renamed: Dict[str, Dict[str, str]]) -> None:
        self._renamed = renamed

    def warn_about(self, config: dict) -> None:
        for section, keys in self._renamed.items():
            section_config = config.get(section)
            if not isinstance(section_config, dict):
                continue
            for old_name, new_name in keys.items():
                if old_name in section_config:
                    LOGGER.warning(
                        f"{section}.{old_name} was renamed to {section}.{new_name} and is being ignored. "
                        f"Rename it to keep its effect.",
                    )


class DeprecatedAdapterKey:
    """params.lora was renamed to params.adapter when DoRA was added, since the same hyperparameters now
    back both. A config using the old name is migrated rather than rejected."""

    def apply_to(self, config: dict) -> None:
        params = config.get("params")
        if isinstance(params, dict) and "lora" in params and "adapter" not in params:
            LOGGER.warning("params.lora is deprecated; rename it to params.adapter.")
            params["adapter"] = params.pop("lora")
