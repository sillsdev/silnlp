import logging
from pathlib import Path

from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    TrainingArguments,
)

from silnlp.common.utils import merge_dict


LOGGER = logging.getLogger(__name__)

TRAINING_ARGS_CONFIG_MAPPING = {
    "train": {
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "group_by_length",
        "log_level",
        "logging_dir",
        "logging_first_step",
        "logging_nan_inf_filter",
        "logging_steps",
        "logging_strategy",
        "max_steps",
        "num_train_epochs",
        "output_dir",
        "per_device_train_batch_size",
        "save_steps",
        "save_strategy",
        "save_total_limit",
    },
    "eval": {
        "eval_steps",
        "eval_strategy",
        "greater_is_better",
        "load_best_model_at_end",
        "metric_for_best_model",
        "per_device_eval_batch_size",
    },
    "params": {
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
    },
}


class HuggingFaceConfig:
    def __init__(self, exp_dir: Path, config: dict) -> None:
        self.exp_dir = exp_dir
        config = merge_dict(
            {
                "data": {
                    "test_size": 0.1,
                    "corpora": [],
                    "characters_to_remove": "",
                    "max_input_length": 30.0,
                },
                "train": {
                    "save_steps": 250,
                    "per_device_train_batch_size": 2,
                    "save_strategy": "steps",
                    "save_total_limit": 2,
                    "gradient_accumulation_steps": 8,
                    "max_steps": 1000,
                    "group_by_length": True,
                    "output_dir": str(exp_dir / "run"),
                },
                "eval": {
                    "eval_strategy": "steps",
                    "eval_steps": 250,
                    "early_stopping": None,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "cer",
                    "per_device_eval_batch_size": 2,
                },
                "params": {
                    "warmup_steps": 100,
                    "learning_rate": 0.0002,
                    "lr_scheduler_type": "constant_with_warmup",
                },
            },
            config,
        )
        self.root = config

    @property
    def model_dir(self) -> Path:
        return Path(self.train["output_dir"])

    @property
    def model(self) -> str:
        return self.root["model"]

    @property
    def params(self) -> dict:
        return self.root["params"]

    @property
    def data(self) -> dict:
        return self.root["data"]

    @property
    def train(self) -> dict:
        return self.root["train"]

    @property
    def infer(self) -> dict:
        return self.root["infer"]

    @property
    def eval(self) -> dict:
        return self.root["eval"]


def create_seq2seq_training_arguments(config: HuggingFaceConfig) -> Seq2SeqTrainingArguments:
    parser = HfArgumentParser(Seq2SeqTrainingArguments)
    args: dict = {}
    for section, params in TRAINING_ARGS_CONFIG_MAPPING.items():
        section_config: dict = config.root[section]
        for param in params:
            if param in section_config:
                args[param] = section_config[param]
    merge_dict(
            args,
            {
                "fp16": True,
                "predict_with_generate": True
            },
        )
    return parser.parse_dict(args)[0]


def create_training_arguments(config: HuggingFaceConfig) -> TrainingArguments:
    parser = HfArgumentParser(TrainingArguments)
    args: dict = {}
    for section, params in TRAINING_ARGS_CONFIG_MAPPING.items():
        section_config: dict = config.root[section]
        for param in params:
            if param in section_config:
                args[param] = section_config[param]
    merge_dict(
        args,
        {
            "fp16": True,
        },
    )
    return parser.parse_dict(args)[0]
