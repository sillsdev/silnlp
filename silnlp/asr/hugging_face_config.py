import gc
import json
import logging
import os
import re
import shutil
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from itertools import repeat
from math import prod
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple, TypeVar, Union, cast

import datasets.utils.logging as datasets_logging
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers.utils.logging as transformers_logging
import yaml
from accelerate.utils.memory import should_reduce_batch_size
from datasets import Dataset
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef
from sacremoses import MosesPunctNormalizer
from tokenizers import AddedToken, NormalizedString, Regex
from tokenizers.implementations import SentencePieceBPETokenizer, SentencePieceUnigramTokenizer
from tokenizers.normalizers import Normalizer
from torch import Tensor, nn, optim
from torch.utils.data import Sampler
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBartTokenizer,
    MBartTokenizerFast,
    NllbTokenizer,
    NllbTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
    T5TokenizerFast,
    TensorType,
    TrainerCallback,
    TranslationPipeline,
    TrainingArguments,
    set_seed,
)
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers.generation import BeamSearchEncoderDecoderOutput, GreedySearchEncoderDecoderOutput
from transformers.modeling_utils import unwrap_model
from transformers.tokenization_utils import BatchEncoding, TruncationStrategy
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    PaddingStrategy,
    is_peft_available,
    is_safetensors_available,
    to_py_obj,
)
from transformers.utils.logging import tqdm

from silnlp.common.utils import merge_dict

from ..common.corpus import Term, count_lines, get_terms
from ..common.environment import SIL_NLP_ENV
from ..common.translator import (
    DraftGroup,
    SentenceTranslation,
    SentenceTranslationGroup,
    generate_test_confidence_files,
)

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
            },
        )
    parser.parse_dict(args)[0]


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
