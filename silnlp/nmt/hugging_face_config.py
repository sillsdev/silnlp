import gc
import json
import logging
import os
import re
import shutil
from contextlib import ExitStack
from copy import deepcopy
from enum import Enum
from itertools import repeat
from math import exp, prod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union, cast

import datasets.utils.logging as datasets_logging
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers.utils.logging as transformers_logging
import yaml
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils.memory import should_reduce_batch_size
from datasets import Dataset
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef
from sacremoses import MosesPunctNormalizer
from tokenizers import AddedToken, NormalizedString, Regex
from tokenizers.implementations import SentencePieceBPETokenizer, SentencePieceUnigramTokenizer
from tokenizers.normalizers import Normalizer
from torch import Tensor, TensorType, nn, optim
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
    TrainerCallback,
    TranslationPipeline,
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

from ..common.corpus import Term, count_lines, get_terms
from ..common.environment import SIL_NLP_ENV
from ..common.translator import DraftGroup, TranslationGroup, generate_confidence_files
from ..common.utils import NoiseMethod, ReplaceRandomToken, Side, create_noise_methods, get_mt_exp_dir, merge_dict
from .config import CheckpointType, Config, DataFile, NMTModel
from .tokenizer import NullTokenizer, Tokenizer

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

LOGGER = logging.getLogger(__name__)


def prepare_decoder_input_ids_from_labels(self: M2M100ForConditionalGeneration, labels: Tensor) -> Tensor:
    # shift ids to the right
    shifted_input_ids = labels.new_zeros(labels.shape)
    shifted_input_ids[:, 1:] = labels[:, :-1].clone()
    assert self.config.decoder_start_token_id is not None
    shifted_input_ids[:, 0] = self.config.decoder_start_token_id

    if self.config.pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.config.pad_token_id)

    return shifted_input_ids


M2M100ForConditionalGeneration.prepare_decoder_input_ids_from_labels = prepare_decoder_input_ids_from_labels

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
        "save_on_each_node",
        "save_steps",
        "save_strategy",
        "save_total_limit",
    },
    "eval": {
        "eval_accumulation_steps",
        "eval_delay",
        "eval_steps",
        "eval_strategy",
        "greater_is_better",
        "include_inputs_for_metrics",
        "load_best_model_at_end",
        "metric_for_best_model",
        "per_device_eval_batch_size",
        "predict_with_generate",
    },
    "params": {
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "full_determinism",
        "generation_max_length",
        "generation_num_beams",
        "label_smoothing_factor",
        "learning_rate",
        "lr_scheduler_type",
        "max_grad_norm",
        "optim",
        "warmup_ratio",
        "warmup_steps",
        "weight_decay",
    },
}

LORA_DEFAULT_CONFIGS = {
    "facebook/nllb-200": {
        "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "modules_to_save": ["embed_tokens", "lm_head"],
    },
    "google/madlad400": {
        "target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
        "modules_to_save": ["embed_tokens", "lm_head"],
    },
}

SP_TOKENIZER_CONFIG = {
    "facebook/nllb-200": {"type": "BPE", "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]},
    "google/madlad400": {"type": "Unigram", "special_tokens": ["<unk>", "<s>", "</s>"], "unk_token": "<unk>"},
}

# "loss" and "eval_loss" are both evaluation loss
# The early stopping callback adds "eval_" to all metrics that don't already start with it
DEFAULT_METRICS = ["loss", "eval_loss"]
EVAL_METRICS_MODULES = {"bleu": "sacrebleu", "chrf3": "chrf", "chrf3+": "chrf", "chrf3++": "chrf", "m-bleu": "sacrebleu", "m-chrf3": "chrf3", "m-chrf3+": "chrf3", "m-chrf3++": "chrf3"}


def get_best_checkpoint(model_dir: Path) -> Path:
    trainer_state_path = model_dir / "trainer_state.json"
    with trainer_state_path.open("r", encoding="utf-8") as f:
        trainer_state = json.load(f)
    return model_dir / Path(trainer_state["best_model_checkpoint"]).name


def has_best_checkpoint(model_dir: Path) -> bool:
    trainer_state_path = model_dir / "trainer_state.json"
    with trainer_state_path.open("r", encoding="utf-8") as f:
        trainer_state = json.load(f)
    return "best_model_checkpoint" in trainer_state and trainer_state["best_model_checkpoint"] is not None


def get_parent_last_checkpoint(model_dir: Path) -> Path:
    trainer_state_path = model_dir / "trainer_state.json"
    with trainer_state_path.open("r", encoding="utf-8") as f:
        trainer_state = json.load(f)
    max_step = trainer_state["max_steps"]
    last_checkpoint = "checkpoint-" + str(max_step)
    return model_dir / last_checkpoint


OPTIMIZER_STATE_FILES = {"optimizer.pt", "rng_state.pth", "scaler.pt", "scheduler.pt"}


def delete_optimizer_state(checkpoint_path: Path) -> None:
    for file in OPTIMIZER_STATE_FILES:
        path = checkpoint_path / file
        if path.is_file():
            path.unlink()


TOKENIZER_FILES = {
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
}


def delete_tokenizer(checkpoint_path: Path) -> None:
    for file in TOKENIZER_FILES:
        path = checkpoint_path / file
        if path.is_file():
            path.unlink()


def add_lang_code_to_tokenizer(tokenizer: PreTrainedTokenizer, lang_code: str) -> None:
    tokenizer.add_special_tokens({"additional_special_tokens": [lang_code]}, replace_additional_special_tokens=False)
    lang_id = tokenizer.convert_tokens_to_ids(lang_code)
    if isinstance(tokenizer, (MBart50Tokenizer, MBartTokenizer)):
        tokenizer.id_to_lang_code[lang_id] = lang_code
        tokenizer.fairseq_tokens_to_ids[lang_code] = lang_id
        tokenizer.fairseq_ids_to_tokens[lang_id] = lang_code
    elif isinstance(tokenizer, M2M100Tokenizer):
        tokenizer.lang_code_to_token[lang_code] = lang_code
        tokenizer.lang_token_to_id[lang_code] = lang_id
        tokenizer.id_to_lang_token[lang_id] = lang_code


def is_sublist(sub: List[int], lst: List[int]) -> bool:
    ln = len(sub)
    if ln >= len(lst):
        return False
    return any(lst[i : i + ln] == sub for i in range(len(sub) - ln + 1))


def prune_sublists(words_ids: List[List[List[int]]]) -> List[List[List[int]]]:
    result: List[List[List[int]]] = []
    for variants in words_ids:
        temp_variants: List[List[int]] = []
        for i in range(len(variants)):
            if not any(is_sublist(variants[i], variants[j]) for j in range(len(variants)) if i != j):
                temp_variants.append(variants[i])
        if len(temp_variants) > 0:
            result.append(temp_variants)
    return result


SUPPORTED_MODEL_PREFIXES = ["facebook/nllb-200", "google/madlad400"]
SUPPORTED_T5_MODELS = ["google/madlad400"]


def get_model_prefix(model: str) -> str:
    for prefix in SUPPORTED_MODEL_PREFIXES:
        if model.startswith(prefix):
            return prefix
    return ""


def get_parent_model_prefix(parent_exp: str) -> str:
    parent_dir = Path(get_mt_exp_dir(parent_exp))
    with (parent_dir / "config.yml").open("r", encoding="utf-8") as file:
        parent_configs = yaml.safe_load(file)
    parent_base_model = parent_configs.get("model")
    parent_model_prefix = get_model_prefix(parent_base_model)
    return parent_model_prefix


def get_parent_model_name(parent_exp: str) -> str:
    parent_dir = Path(get_mt_exp_dir(parent_exp))
    parent_model_dir = parent_dir / "run"
    parent_model = get_parent_last_checkpoint(parent_model_dir)
    if has_best_checkpoint(parent_model_dir):
        parent_model = get_best_checkpoint(parent_model_dir)
    LOGGER.info("Using parent model. This might be different from the model specified in config.")
    return str(parent_model)


class HuggingFaceConfig(Config):
    def __init__(self, exp_dir: Path, config: dict) -> None:
        ckpt_dir = str(exp_dir / "run") if config["use_default_model_dir"] else SIL_NLP_ENV.get_temp_model_dir()
        config = merge_dict(
            {
                "data": {
                    "mirror": False,
                    "seed": 111,
                    "tokenize": True,
                    "aligner": "fast_align",
                    "stats_max_size": 100000,  # a little over the size of the bible
                    "terms": {"train": True, "categories": "PN", "include_glosses": True, "dictionary": False},
                    "lang_codes": {},
                    "add_new_lang_code": True,
                },
                "train": {
                    "max_source_length": 200,
                    "max_target_length": 200,
                    "gradient_checkpointing": True,
                    "gradient_checkpointing_kwargs": {"use_reentrant": True},
                    "save_steps": 1000,
                    "per_device_train_batch_size": 16,
                    "save_strategy": "steps",
                    "save_total_limit": 2,
                    "gradient_accumulation_steps": 4,
                    "auto_grad_acc": False,
                    "max_steps": 5000,
                    "group_by_length": True,
                    "output_dir": ckpt_dir,
                    "delete_checkpoint_optimizer_state": True,
                    "delete_checkpoint_tokenizer": True,
                    "log_level": "info",
                    "use_lora": False,
                    "lora_config": {},
                },
                "eval": {
                    "eval_strategy": "steps",
                    "eval_steps": 1000,
                    "early_stopping": None,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "bleu",
                    "per_device_eval_batch_size": 16,
                    "multi_ref_eval": False,
                    "predict_with_generate": True,
                    "detokenize": True,
                },
                "infer": {
                    "infer_batch_size": 16,
                    "num_beams": 2,
                    "num_drafts": 3,
                    "multiple_translations_method": "hybrid",
                    "temperature": 0.75,
                    "diversity_penalty": 1.0,
                },
                "params": {
                    "optim": "adamw_torch",
                    "label_smoothing_factor": 0.2,
                    "warmup_steps": 1000,
                    "dropout": 0.1,
                    "attention_dropout": 0.1,
                    "activation_dropout": 0.0,
                    "learning_rate": 0.0002,
                    "lr_scheduler_type": "cosine",
                    "attn_implementation": "sdpa",
                },
                "model": "facebook/nllb-200-distilled-1.3B",
            },
            config,
        )
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_prefix = get_model_prefix(config.get("model", ""))

        if "parent" in config["data"]:
            parent = config["data"]["parent"]
            parent_model_name = get_parent_model_name(parent)
            parent_model_prefix = get_parent_model_prefix(parent)
            if parent_model_prefix != self.model_prefix:
                LOGGER.error("The parent model and the config model are not in the same type.")
                raise ValueError(f"Unmatched model prefix {parent_model_prefix} and {self.model_prefix}")
            config["model"] = parent_model_name
            self.model_prefix = parent_model_prefix

        super().__init__(exp_dir, config)

        if self.model_prefix == "google/madlad400":
            self.train["max_source_length"] = 256
            self.train["max_target_length"] = 256

        # disable evaluation if there is no validation split
        if not self.has_val_split:
            config["eval"]["eval_strategy"] = "no"
            config["eval"]["load_best_model_at_end"] = False
            config["eval"]["early_stopping"] = None
            config["eval"]["metric_for_best_model"] = None

        if config["train"]["auto_grad_acc"]:
            config["train"]["per_device_train_batch_size"] = 64
            config["train"]["gradient_accumulation_steps"] = 1

    @property
    def model_dir(self) -> Path:
        return Path(self.train["output_dir"])

    @property
    def val_src_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.default_val_src_iso, self.default_val_src_iso)

    @property
    def test_src_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.default_test_src_iso, self.default_test_src_iso)

    @property
    def val_trg_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.default_val_trg_iso, self.default_val_trg_iso)

    @property
    def test_trg_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.default_test_trg_iso, self.default_test_trg_iso)

    @property
    def has_best_checkpoint(self) -> bool:
        return has_best_checkpoint(self.model_dir)

    def create_model(self, mixed_precision: bool = True, num_devices: int = 1) -> NMTModel:
        return HuggingFaceNMTModel(self, mixed_precision, num_devices)

    def create_tokenizer(self) -> Tokenizer:
        if not self.data["tokenize"]:
            return NullTokenizer()
        return HuggingFaceTokenizer(
            self.get_tokenizer(),
            self.data["lang_codes"],
            self.train["max_source_length"],
            self.train["max_target_length"],
        )

    def _add_tokens(
        self,
        missing_tokens: List[str],
        trained_tokenizers: Optional[List[Union[SentencePieceBPETokenizer, SentencePieceUnigramTokenizer]]] = None,
    ) -> None:
        assert self._tokenizer is not None
        self._tokenizer.save_pretrained(str(self.exp_dir))
        with open(self.exp_dir / "tokenizer.json", "r+", encoding="utf-8") as file:
            data = json.load(file)
            if data["model"]["type"] == "BPE":
                vocab_len = len(data["model"]["vocab"].keys())
                for i, token in enumerate(missing_tokens):
                    data["model"]["vocab"][token] = vocab_len + i
                if trained_tokenizers:
                    for trained_tok in trained_tokenizers:
                        trained_tok.save(str(self.exp_dir / "tokenizer_trained.json"))
                        with open(self.exp_dir / "tokenizer_trained.json", "r+", encoding="utf-8") as trained_file:
                            trained_data = json.load(trained_file)
                            data["model"]["merges"] = trained_data["model"]["merges"] + data["model"]["merges"]
            elif data["model"]["type"] == "Unigram":
                if trained_tokenizers:
                    for trained_tok in trained_tokenizers:
                        trained_tok.save(str(self.exp_dir / "tokenizer_trained.json"))
                        with open(self.exp_dir / "tokenizer_trained.json", "r+", encoding="utf-8") as trained_file:
                            trained_data = json.load(trained_file)
                            # Use the probability from the base tokenizer for tokens already in the base tokenizer
                            base_toks = [t[0] for t in data["model"]["vocab"]]
                            for i in reversed(range(len(trained_data["model"]["vocab"]))):
                                if trained_data["model"]["vocab"][i][0] in base_toks:
                                    del trained_data["model"]["vocab"][i]
                            data["model"]["vocab"] = data["model"]["vocab"] + trained_data["model"]["vocab"]
                else:
                    for token in missing_tokens:
                        data["model"]["vocab"].append([token, -18])
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.truncate()
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.exp_dir), use_fast=True)
        return

    def _train_sp_tokenizer(self, files, vocab_size) -> Union[SentencePieceBPETokenizer, SentencePieceUnigramTokenizer]:
        assert self._tokenizer is not None
        sp_tok_config = SP_TOKENIZER_CONFIG[self.model_prefix]
        sp_tok = SentencePieceBPETokenizer() if sp_tok_config["type"] == "BPE" else SentencePieceUnigramTokenizer()
        hf_tokenizer = HuggingFaceTokenizer(
            self._tokenizer, self.data["lang_codes"], self.train["max_source_length"], self.train["max_target_length"]
        )
        sp_tok.normalizer = Normalizer.custom(CustomNormalizerWrapper(hf_tokenizer))

        if sp_tok_config["type"] == "BPE":
            sp_tok.train(files, vocab_size=vocab_size, min_frequency=2, special_tokens=sp_tok_config["special_tokens"])
        elif sp_tok_config["type"] == "Unigram":
            sp_tok: SentencePieceUnigramTokenizer
            sp_tok.train(
                files,
                vocab_size=vocab_size,
                special_tokens=sp_tok_config["special_tokens"],
                unk_token=sp_tok_config["unk_token"],
            )
        sp_tok.normalizer = self._tokenizer.backend_tokenizer.normalizer
        return sp_tok

    def _create_trained_tokens(
        self, file_paths, vocab_size
    ) -> Tuple[List[str], Union[SentencePieceBPETokenizer, SentencePieceUnigramTokenizer]]:
        assert self._tokenizer is not None
        files = [str(f) for f in file_paths]
        sp_tokenizer = self._train_sp_tokenizer(files, vocab_size)
        sp_keys, tok_keys = sp_tokenizer.get_vocab().keys(), self._tokenizer.get_vocab().keys()
        missing_tokens = sorted(list(set(sp_keys) - set(tok_keys)))
        return missing_tokens, sp_tokenizer

    def _find_missing_characters(self, corpus: List[Path]) -> List[str]:
        assert self._tokenizer is not None
        vocab = self._tokenizer.get_vocab().keys()
        charset: Set[str] = set()
        hf_tokenizer = HuggingFaceTokenizer(
            self._tokenizer, self.data["lang_codes"], self.train["max_source_length"], self.train["max_target_length"]
        )
        for file in corpus:
            with file.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    charset = charset | set(hf_tokenizer.normalize(Side.TARGET, line))

        charset = set(filter(None, {char.strip() for char in charset}))
        missing_characters = sorted(list(charset - vocab))
        return missing_characters

    def _build_vocabs(self, stats: bool = False) -> None:
        tok_dict = self.data.get("tokenizer")
        self._tokenizer = self.get_or_create_tokenizer()
        trained_tokenizers = []
        missing_tokens: List[str] = []
        src_missing_tokens: List[str] = []
        trg_missing_tokens: List[str] = []
        if tok_dict and (tok_dict.get("update_src") or tok_dict.get("update_trg")):
            if (
                tok_dict.get("trained_tokens")
                and (SIL_NLP_ENV.assets_dir / "tokenizers" / self.model_prefix / "tokenizer_config.json").is_file()
            ):
                if not tok_dict.get("share_vocab") and tok_dict.get("update_src") and tok_dict.get("update_trg"):
                    src_missing_tokens, src_trained_tokenizer = self._create_trained_tokens(
                        list(self.src_file_paths), tok_dict.get("src_vocab_size")
                    )
                    trg_missing_tokens, trg_trained_tokenizer = self._create_trained_tokens(
                        list(self.trg_file_paths), tok_dict.get("trg_vocab_size")
                    )
                    trg_missing_tokens = sorted(list(set(trg_missing_tokens) - set(src_missing_tokens)))
                    missing_tokens = src_missing_tokens + trg_missing_tokens
                    trained_tokenizers = [src_trained_tokenizer] + [trg_trained_tokenizer]
                else:
                    if tok_dict.get("share_vocab") and tok_dict.get("update_src") and tok_dict.get("update_trg"):
                        missing_tokens, trained_tokenizer = self._create_trained_tokens(
                            list(self.src_file_paths) + list(self.trg_file_paths),
                            tok_dict.get("src_vocab_size") + tok_dict.get("trg_vocab_size"),
                        )
                    elif tok_dict.get("update_src"):
                        missing_tokens, trained_tokenizer = self._create_trained_tokens(
                            list(self.src_file_paths), tok_dict.get("src_vocab_size")
                        )
                        src_missing_tokens = missing_tokens
                    elif tok_dict.get("update_trg"):
                        missing_tokens, trained_tokenizer = self._create_trained_tokens(
                            list(self.trg_file_paths), tok_dict.get("trg_vocab_size")
                        )
                        trg_missing_tokens = missing_tokens
                    trained_tokenizers.append(trained_tokenizer)
            else:
                if tok_dict.get("update_src"):
                    missing_tokens = src_missing_tokens = self._find_missing_characters(list(self.src_file_paths))
                if tok_dict.get("update_trg"):
                    missing_tokens = trg_missing_tokens = self._find_missing_characters(list(self.trg_file_paths))
                if tok_dict.get("update_src") and tok_dict.get("update_trg"):
                    trg_missing_tokens = sorted(list(set(trg_missing_tokens) - set(src_missing_tokens)))
                    missing_tokens = src_missing_tokens + trg_missing_tokens

            if missing_tokens:
                self._add_tokens(missing_tokens, trained_tokenizers)

            if tok_dict.get("share_vocab") and tok_dict.get("update_src") and tok_dict.get("update_trg"):
                # TODO: Calculate representative split of tokens for shared vocab case
                stats_data = [
                    ["Source", int(len(missing_tokens) / 2)],
                    ["Target", len(missing_tokens) - int(len(missing_tokens) / 2)],
                ]
            else:
                stats_data = [
                    ["Source", len(src_missing_tokens)],
                    ["Target", len(trg_missing_tokens)],
                ]
        else:
            stats_data = [
                ["Source", 0],
                ["Target", 0],
            ]

        if stats and self.data["tokenize"]:
            stats_columns = pd.MultiIndex.from_tuples(
                [
                    (" ", "Translation Side"),
                    (" ", "Num Tokens Added to Vocab"),
                ]
            )
            stats_df = pd.DataFrame(stats_data, columns=stats_columns)
            stats_df.to_csv(self.exp_dir / "tokenization_stats.csv", index=False)
            stats_df.to_excel(self.exp_dir / "tokenization_stats.xlsx")

        if self.data["add_new_lang_code"]:
            lang_codes: Dict[str, str] = self.data["lang_codes"]
            updated = False
            for iso in self.src_isos | self.trg_isos:
                lang_code = lang_codes.get(iso, iso)
                if isinstance(self._tokenizer, (T5Tokenizer, T5TokenizerFast)):
                    if lang_code not in self._tokenizer.all_special_tokens and iso in self.trg_isos:
                        add_lang_code_to_tokenizer(self._tokenizer, lang_code)
                        updated = True
                elif isinstance(self._tokenizer, (MBartTokenizer, MBartTokenizerFast)):
                    if lang_code not in self._tokenizer.lang_code_to_id:
                        add_lang_code_to_tokenizer(self._tokenizer, lang_code)
                        updated = True
                elif isinstance(self._tokenizer, (NllbTokenizer, NllbTokenizerFast)):
                    add_lang_code_to_tokenizer(self._tokenizer, lang_code)
                    updated = True
                elif lang_code not in self._tokenizer.lang_code_to_id:
                    add_lang_code_to_tokenizer(self._tokenizer, lang_code)
                    updated = True
            if updated:
                self._tokenizer.save_pretrained(self.exp_dir)

        if len(self._tags) > 0:
            self._tokenizer.add_tokens([AddedToken(tag, rstrip=True, special=True) for tag in self._tags])

    def get_or_create_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            tok_dict = self.data.get("tokenizer")
            if (
                tok_dict
                and (tok_dict.get("update_src") or tok_dict.get("update_trg"))
                and ((self.exp_dir / "sentencepiece.bpe.model").is_file() or (self.exp_dir / "spiece.model").is_file())
                and not (self.exp_dir / "tokenizer_config.json").is_file()
            ):
                if self.model_prefix == "facebook/nllb-200":
                    self._tokenizer = NllbTokenizer.from_pretrained(str(self.exp_dir))
                    self._tokenizer = convert_slow_tokenizer(self._tokenizer)
                    self._tokenizer = NllbTokenizerFast(tokenizer_object=self._tokenizer)
                    self._tokenizer.save_pretrained(str(self.exp_dir))
                elif self.model_prefix == "google/madlad400":
                    self._tokenizer = T5Tokenizer.from_pretrained(str(self.exp_dir))
                    self._tokenizer = convert_slow_tokenizer(self._tokenizer)
                    self._tokenizer = T5TokenizerFast(tokenizer_object=self._tokenizer)
                    self._tokenizer.add_special_tokens(
                        {"additional_special_tokens": ["<s>"]}, replace_additional_special_tokens=False
                    )
                    self._tokenizer.save_pretrained(str(self.exp_dir))
            else:
                if (not tok_dict or not (tok_dict.get("update_src") or tok_dict.get("update_trg"))) and (
                    self.exp_dir / "tokenizer_config.json"
                ).is_file():
                    model_name_or_path = str(self.exp_dir)
                elif (tok_dict and (tok_dict.get("update_src") or tok_dict.get("update_trg"))) and (
                    SIL_NLP_ENV.assets_dir / "tokenizers" / self.model_prefix / "tokenizer_config.json"
                ).is_file():
                    model_name_or_path = str(SIL_NLP_ENV.assets_dir / "tokenizers" / self.model_prefix)
                elif self.has_parent:
                    parent_exp = self.data["parent"]
                    parent_dir = Path(get_mt_exp_dir(parent_exp))
                    model_name_or_path = str(parent_dir)
                else:
                    model_name_or_path = self.model
                self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            self._tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return self._tokenizer

    def get_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            if (self.exp_dir / "tokenizer_config.json").is_file():
                model_name_or_path = str(self.exp_dir)
            elif self.has_parent:
                parent_exp = self.data["parent"]
                parent_dir = Path(get_mt_exp_dir(parent_exp))
                model_name_or_path = str(parent_dir)
            else:
                model_name_or_path = self.model

            self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            self._tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return self._tokenizer

    def _write_dictionary(
        self,
        tokenizer: Tokenizer,
        src_terms_files: List[Tuple[DataFile, str]],
        trg_terms_files: List[Tuple[DataFile, str]],
    ) -> int:
        terms_config = self.data["terms"]
        dict_count = 0
        with ExitStack() as stack:
            dict_trg_file = stack.enter_context(self._open_append(self.dict_trg_filename()))
            dict_vref_file = stack.enter_context(self._open_append(self.dict_vref_filename()))

            categories: Optional[Union[str, List[str]]] = terms_config["categories"]
            if isinstance(categories, str):
                categories = [cat.strip() for cat in categories.split(",")]
            if categories is not None and len(categories) == 0:
                return 0
            categories_set: Optional[Set[str]] = None if categories is None else set(categories)

            if terms_config["include_glosses"]:
                gloss_iso: str = str(terms_config["include_glosses"]).lower()
                if gloss_iso == "true":
                    src_gloss_iso = list(self.src_isos.intersection(["en", "fr", "id", "es"]))
                    trg_gloss_iso = list(self.trg_isos.intersection(["en", "fr", "id", "es"]))
                    if src_gloss_iso:
                        gloss_iso = src_gloss_iso[0]
                    elif trg_gloss_iso:
                        gloss_iso = trg_gloss_iso[0]
                    else:
                        LOGGER.warning(
                            "Glosses could not be included. No source or target language matches any of the supported gloss language codes: en, fr, id, es."
                        )
                        gloss_iso = None
                elif gloss_iso not in ["en", "fr", "id", "es"]:
                    LOGGER.warning(
                        f"Gloss language code, {gloss_iso}, does not match the supported gloss language codes: en, fr, id, es."
                    )
                    gloss_iso = None
            else:
                gloss_iso = None

            all_trg_terms: List[Tuple[DataFile, Dict[str, Term], str]] = []
            for trg_terms_file, tags_str in trg_terms_files:
                all_trg_terms.append((trg_terms_file, get_terms(trg_terms_file.path, iso=gloss_iso), tags_str))
            for trg_terms_file, trg_terms, tags_str in all_trg_terms:
                tokenizer.set_trg_lang(trg_terms_file.iso)
                for trg_term in trg_terms.values():
                    if categories_set is not None and trg_term.cat not in categories_set:
                        continue

                    renderings: List[str] = []
                    for rendering in trg_term.renderings:
                        renderings.append(
                            tokenizer.tokenize(Side.TARGET, rendering, add_dummy_prefix=True, add_special_tokens=False)
                        )
                        renderings.append(
                            tokenizer.tokenize(Side.TARGET, rendering, add_dummy_prefix=False, add_special_tokens=False)
                        )
                    if len(renderings) == 0:
                        continue
                    dict_trg_file.write("\t".join(renderings) + "\n")
                    dict_vref_file.write("\t".join(str(vref) for vref in trg_term.vrefs) + "\n")
                    dict_count += 1

            if gloss_iso is not None:
                all_src_terms: List[Tuple[DataFile, Dict[str, Term], str]] = []
                for src_terms_file, tags_str in src_terms_files:
                    all_src_terms.append((src_terms_file, get_terms(src_terms_file.path, iso=gloss_iso), tags_str))
                tokenizer.set_trg_lang(gloss_iso)
                for src_term_file, src_terms, tags_str in all_src_terms:
                    for src_term in src_terms.values():
                        if categories_set is not None and src_term.cat not in categories_set:
                            continue

                        glosses: List[str] = []
                        for gloss in src_term.glosses:
                            glosses.append(
                                tokenizer.tokenize(Side.TARGET, gloss, add_dummy_prefix=True, add_special_tokens=False)
                            )
                            glosses.append(
                                tokenizer.tokenize(Side.TARGET, gloss, add_dummy_prefix=False, add_special_tokens=False)
                            )
                        if len(glosses) == 0:
                            continue
                        dict_trg_file.write("\t".join(glosses) + "\n")
                        dict_vref_file.write("\t".join(str(vref) for vref in src_term.vrefs) + "\n")
                        dict_count += 1
        return dict_count


def batch_prepare_for_model(
    tokenizer: PreTrainedTokenizer,
    batch_tokens: List[List[str]],
    return_tensors: Optional[Union[str, TensorType]] = None,
) -> BatchEncoding:
    batch_outputs: Dict[str, Any] = {}
    for tokens in batch_tokens:
        ids = tokenizer.convert_tokens_to_ids(tokens)
        outputs = tokenizer.prepare_for_model(ids, add_special_tokens=False)

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)
    return BatchEncoding(batch_outputs, tensor_type=return_tensors)


TSent = TypeVar("TSent")


def batch_sentences(
    sentences: Iterable[TSent],
    vrefs: Optional[Iterable[VerseRef]],
    batch_size: int,
    dictionary: Dict[VerseRef, Set[str]],
) -> Iterable[Tuple[List[TSent], Optional[List[List[List[str]]]]]]:
    batch: List[TSent] = []
    for sentence, vref in zip(sentences, repeat(None) if vrefs is None else vrefs):
        terms: Set[str] = set()
        if vref is not None:
            for vr in vref.all_verses():
                terms.update(dictionary.get(vr, set()))
        if len(terms) > 0:
            if len(batch) > 0:
                yield batch, None
                batch = []
            force_words = [[term.split() for term in term.split("\t")] for term in terms]
            yield [sentence], force_words
        else:
            batch.append(sentence)
            if len(batch) == batch_size:
                yield batch, None
                batch = []
    if len(batch) > 0:
        yield batch, None


class OutputGroup:
    def __init__(self, outputs: List[dict]):
        self.outputs = outputs

    def get_translated_text(self) -> List[str]:
        return [output["translation_text"] for output in self.outputs]

    def get_token_ids(self) -> List[List[int]]:
        return [output["translation_token_ids"] for output in self.outputs]

    def get_token_scores(self) -> List[float]:
        return [output["token_scores"] for output in self.outputs]

    def get_sequence_score(self) -> List[float]:
        return [output["sequence_score"] for output in self.outputs]


class HuggingFaceNMTModel(NMTModel):
    def __init__(self, config: HuggingFaceConfig, mixed_precision: bool, num_devices: int) -> None:
        self._config = config
        self._mixed_precision = mixed_precision
        set_seed(self._config.data["seed"])
        self._dictionary: Optional[Dict[VerseRef, Set[str]]] = None
        self._is_t5 = self._config.model_prefix in SUPPORTED_T5_MODELS
        self._num_devices = num_devices

    def train(self) -> None:
        training_args = self._create_training_arguments()

        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers_logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        datasets_logging.set_verbosity(log_level)
        transformers_logging.set_verbosity(log_level)
        transformers_logging.enable_default_handler()
        transformers_logging.enable_explicit_format()

        model_config = AutoConfig.from_pretrained(
            self._config.model,
            use_cache=not training_args.gradient_checkpointing,
            dropout=self._config.params["dropout"],
            attention_dropout=self._config.params["attention_dropout"],
            activation_dropout=self._config.params["activation_dropout"],
            label2id={},
            id2label={},
            num_labels=0,
            attn_implementation=self._config.params["attn_implementation"],
        )
        if self._num_devices == 2 and self._config.model_prefix == "facebook/nllb-200":
            device_map = {
                "lm_head": 0,
                "model.shared": 0,
                "model.encoder": 0,
                "model.decoder.embed_tokens": 0,
                "model.decoder.embed_positions": 1,
                "model.decoder.layers": 1,
                "model.decoder.layer_norm": 1,
            }
        else:
            device_map = None
        model = cast(
            PreTrainedModel,
            AutoModelForSeq2SeqLM.from_pretrained(self._config.model, config=model_config, device_map=device_map),
        )
        if self._config.train.get("better_transformer"):
            model = model.to_bettertransformer()
        tokenizer = self._config.get_tokenizer()

        old_embeddings = model.get_input_embeddings()
        old_num_tokens = old_embeddings.weight.size(dim=0)
        tok_dict = self._config.data.get("tokenizer")
        if len(tokenizer) > old_num_tokens and tok_dict is not None and tok_dict.get("init_unk"):
            vocab = tokenizer.get_vocab()
            unk_embedding = old_embeddings.weight.data[vocab["<unk>"]]
            model.resize_token_embeddings(
                len(tokenizer), pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None
            )
            embeddings = model.get_input_embeddings()
            embeddings.weight.data[old_num_tokens:, :] = unk_embedding
            model.tie_weights()
        elif len(tokenizer) != old_num_tokens:
            model.resize_token_embeddings(
                len(tokenizer), pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None
            )

        if self._config.train["use_lora"]:
            model = self._convert_to_lora_model(model)

        # Change specific variables based on the type of model
        model, tokenizer = self._configure_model(
            model,
            tokenizer,
            self._config.val_src_lang if self._config.val_src_lang else self._config.test_src_lang,
            self._config.val_trg_lang if self._config.val_trg_lang else self._config.test_trg_lang,
        )

        def load_text_dataset(src_path: Path, trg_path: Path) -> Optional[Dataset]:
            if not src_path.is_file() or not trg_path.is_file():
                return None
            data = []
            with (
                open(src_path, "r", encoding="utf-8-sig") as src_file,
                open(trg_path, "r", encoding="utf-8-sig") as trg_file,
            ):
                for src_line, trg_line in zip(src_file, trg_file):
                    data.append({"src": src_line.strip(), "trg": trg_line.strip()})
            return Dataset.from_dict({"translation": data})

        train_dataset = load_text_dataset(
            self._config.exp_dir / self._config.train_src_filename(),
            self._config.exp_dir / self._config.train_trg_filename(),
        )

        eval_dataset = load_text_dataset(
            self._config.exp_dir / self._config.val_src_filename(),
            self._config.exp_dir / self._config.val_trg_filename(),
        )

        def encode(examples: dict) -> dict:
            inputs = [ex["src"].split() for ex in examples["translation"]]
            model_inputs = batch_prepare_for_model(tokenizer, inputs)

            targets = [ex["trg"].split() for ex in examples["translation"]]
            labels = batch_prepare_for_model(tokenizer, targets)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        if train_dataset is not None:
            with training_args.main_process_first(desc="train dataset map encoding"):
                train_dataset = train_dataset.map(
                    encode,
                    batched=True,
                    remove_columns=train_dataset.column_names,
                    desc="Encoding train dataset",
                )

        if eval_dataset is not None:
            with training_args.main_process_first(desc="validation dataset map encoding"):
                eval_dataset = eval_dataset.map(
                    encode,
                    batched=True,
                    remove_columns=eval_dataset.column_names,
                    desc="Encoding validation dataset",
                )

        src_noise = create_noise_methods(self._config.train.get("src_noise", []))
        for noise_method in src_noise:
            if isinstance(noise_method, ReplaceRandomToken):
                noise_method.filler_token = tokenizer.convert_tokens_to_ids(noise_method.filler_token)

        data_collator = DataCollatorForSeq2SeqNoising(
            tokenizer,
            model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
            src_noise=src_noise,
        )

        metric_name = ""
        if self._config.eval["metric_for_best_model"] is not None:
            metric_name = self._config.eval["metric_for_best_model"].lower()
            if metric_name not in DEFAULT_METRICS:
                metric_module = EVAL_METRICS_MODULES.get(metric_name)
                if metric_module is None:
                    raise ValueError(f"{metric_name} is not a supported metric.")
                metric = evaluate.load(metric_module)
        all_special_ids = set(tokenizer.all_special_ids)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]

            # Replace -100 in the labels as we can't decode them.
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if self._config.eval["detokenize"]:
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                # Some simple post-processing
                decoded_preds = [pred.strip() for pred in decoded_preds]
                decoded_labels = [[label.strip()] for label in decoded_labels]
            else:
                decoded_preds = [
                    " ".join(
                        tokenizer.convert_ids_to_tokens(int(id)) for id in pred if id not in all_special_ids
                    ).strip()
                    for pred in preds
                ]

                decoded_labels = [
                    [
                        " ".join(
                            tokenizer.convert_ids_to_tokens(int(id)) for id in label if id not in all_special_ids
                        ).strip()
                    ]
                    for label in labels
                ]

            if metric_name == "bleu":
                result = metric.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    lowercase=True,
                    force=not self._config.eval["detokenize"],
                )
            elif metric_module == "chrf":
                result = metric.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    char_order=6,
                    word_order=metric_name.count("+"),
                    beta=3,
                    lowercase=True,
                    eps_smoothing="+" in metric_name,
                )
            result = {metric_name: result["score"]}

            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

        trainer = SilSeq2SeqTrainer(
            model,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics=None if metric_name in DEFAULT_METRICS else compute_metrics,
            sequential_sampling=self._config.train.get("sequential_sampling", False),
            better_transformer=self._config.train.get("better_transformer", False),
            auto_grad_acc=self._config.train.get("auto_grad_acc", False),
            model_prefix=self._config.model_prefix,
        )
        early_stopping: Optional[dict] = self._config.eval["early_stopping"]
        if early_stopping:
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping["steps"],
                    early_stopping_threshold=early_stopping["min_improvement"],
                )
            )
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset) if train_dataset is not None else 0

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        delete_checkpoint_optimizer_state = self._config.train["delete_checkpoint_optimizer_state"]
        delete_checkpoint_tokenizer = self._config.train["delete_checkpoint_tokenizer"]
        delete_checkpoint_adapter = self._config.train["use_lora"] and self._config.model_prefix == "facebook/nllb-200"
        if delete_checkpoint_optimizer_state or delete_checkpoint_tokenizer or delete_checkpoint_adapter:
            for child in Path(training_args.output_dir).iterdir():
                if child.is_dir() and child.name.startswith("checkpoint-"):
                    if delete_checkpoint_optimizer_state:
                        delete_optimizer_state(child)
                    if delete_checkpoint_tokenizer:
                        delete_tokenizer(child)
                    if delete_checkpoint_adapter:
                        self._merge_and_delete_adapter(child, len(tokenizer), training_args.save_safetensors)

    def save_effective_config(self, path: Path) -> None:
        training_args = self._create_training_arguments()
        config = deepcopy(self._config.root)
        for section, params in TRAINING_ARGS_CONFIG_MAPPING.items():
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

    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        vref_paths: Optional[List[Path]] = None,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        tokenizer = self._config.get_tokenizer()
        model = self._create_inference_model(ckpt, tokenizer, self._config.test_src_lang, self._config.test_trg_lang)
        pipeline = PretokenizedTranslationPipeline(
            model=model,
            tokenizer=tokenizer,
            src_lang=self._config.test_src_lang,
            tgt_lang=self._config.test_trg_lang,
            device=0,
        )
        pipeline.model = torch.compile(pipeline.model)
        for input_path, translation_path, vref_path in zip(
            input_paths,
            translation_paths,
            cast(Iterable[Optional[Path]], repeat(None) if vref_paths is None else vref_paths),
        ):
            length = count_lines(input_path)
            with ExitStack() as stack:
                src_file = stack.enter_context(input_path.open("r", encoding="utf-8-sig"))
                sentences = (line.strip().split() for line in src_file)
                vrefs: Optional[Iterable[VerseRef]] = None
                if vref_path is not None:
                    vref_file = stack.enter_context(vref_path.open("r", encoding="utf-8-sig"))
                    vrefs = (VerseRef.from_string(line.strip(), ORIGINAL_VERSIFICATION) for line in vref_file)
                output = list(
                    self._translate_test_sentences(
                        tokenizer, pipeline, sentences, vrefs, length, produce_multiple_translations
                    )
                )
                draft_group = DraftGroup([translation for translation, _, _, _ in output])

                for draft_index, translated_draft in enumerate(draft_group.get_drafts(), 1):
                    if produce_multiple_translations:
                        translation_draft_path = translation_path.with_suffix(
                            f".{draft_index}{translation_path.suffix}"
                        )
                    else:
                        translation_draft_path = translation_path
                    out_file = stack.enter_context(translation_draft_path.open("w", encoding="utf-8", newline="\n"))
                    out_file.write("\n".join(translated_draft) + "\n")

                    if save_confidences:
                        generate_confidence_files(
                            output,
                            translation_path,
                            produce_multiple_translations=produce_multiple_translations,
                            draft_index=draft_index,
                        )

    def _translate_test_sentences(
        self,
        tokenizer: PreTrainedTokenizer,
        pipeline: TranslationPipeline,
        sentences: Iterable[List[str]],
        vrefs: Iterable[VerseRef],
        length: int,
        produce_multiple_translations: bool = False,
    ) -> Iterable[TranslationGroup]:
        num_drafts = self.get_num_drafts()
        if produce_multiple_translations and num_drafts > 1:
            LOGGER.info("Producing %i translated drafts", num_drafts)
        elif produce_multiple_translations and num_drafts <= 1:
            LOGGER.warning(
                "num_drafts must be greater than 1 when using --multiple-translations. "
                "Falling back to a single translation."
            )

        for output_group in tqdm(
            self._translate_sentences(
                tokenizer, pipeline, sentences, vrefs, produce_multiple_translations, return_tensors=True
            ),
            total=length,
            unit="ex",
        ):
            all_ids = to_py_obj(output_group.get_token_ids())
            all_scores = to_py_obj(output_group.get_token_scores())
            sequence_score = to_py_obj(output_group.get_sequence_score())
            ids = []
            token_scores = []
            for output_id, output_score in zip(all_ids, all_scores):
                output_ids = []
                output_scores = []
                for id, score in zip(output_id[1:], output_score[1:]):
                    if id == tokenizer.pad_token_id:
                        continue
                    output_ids.append(id)
                    output_scores.append(score)
                ids.append(output_ids)
                token_scores.append(output_scores)
            # ids = [[id for id in output[1:] if id != tokenizer.pad_token_id] for output in ids]
            tokens = [tokenizer.convert_ids_to_tokens(id_group) for id_group in ids]
            yield [" ".join(token_group) for token_group in tokens], tokens, token_scores, sequence_score

    def get_num_drafts(self) -> int:
        num_drafts = self._config.infer.get("num_drafts", 1)
        return num_drafts

    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        vrefs: Optional[Iterable[VerseRef]] = None,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Iterable[TranslationGroup]:
        src_lang = self._config.data["lang_codes"].get(src_iso, src_iso)
        trg_lang = self._config.data["lang_codes"].get(trg_iso, trg_iso)
        tokenizer = self._config.get_tokenizer()
        model = self._create_inference_model(ckpt, tokenizer, src_lang, trg_lang)
        if model.config.max_length is not None and model.config.max_length < 512:
            model.config.max_length = 512

        # The tokenizer isn't wrapped until after calling _create_inference_model,
        # because the tokenizer's input/output language codes are set there
        if isinstance(tokenizer, (NllbTokenizer, NllbTokenizerFast)):
            tokenizer = PunctuationNormalizingTokenizer(tokenizer)

        pipeline = SilTranslationPipeline(
            model=model,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=trg_lang,
            device=0,
        )

        num_drafts = self.get_num_drafts()
        if produce_multiple_translations and num_drafts > 1:
            LOGGER.info("Producing %i translated drafts", num_drafts)
        elif produce_multiple_translations and num_drafts <= 1:
            LOGGER.warning(
                "num_drafts must be greater than 1 when using --multiple-translations. "
                "Falling back to a single translation."
            )

        pipeline.model = torch.compile(pipeline.model)
        if not isinstance(sentences, list):
            sentences = list(sentences)
        for outputs in tqdm(
            self._translate_sentences(tokenizer, pipeline, sentences, vrefs, produce_multiple_translations),
            total=len(sentences),
            unit="ex",
        ):
            if isinstance(outputs, OutputGroup):
                outputs = [outputs]
            for output_group in outputs:
                translated_text = to_py_obj(output_group.get_translated_text())
                all_ids = to_py_obj(output_group.get_token_ids())
                all_scores = to_py_obj(output_group.get_token_scores())
                sequence_score = to_py_obj(output_group.get_sequence_score())
                ids = []
                token_scores = []
                for output_id, output_score in zip(all_ids, all_scores):
                    output_ids = []
                    output_scores = []
                    for id, score in zip(output_id[1:], output_score[1:]):
                        if id == tokenizer.pad_token_id:
                            continue
                        output_ids.append(id)
                        output_scores.append(score)
                    ids.append(output_ids)
                    token_scores.append(output_scores)
                tokens = [tokenizer.convert_ids_to_tokens(id_group) for id_group in ids]
                yield translated_text, tokens, token_scores, sequence_score

    def get_checkpoint_path(self, ckpt: Union[CheckpointType, str, int]) -> Tuple[Path, int]:
        step: Optional[int] = None
        if isinstance(ckpt, str):
            ckpt = ckpt.lower()
            if "avg" in ckpt:
                ckpt = CheckpointType.AVERAGE
            elif "best" in ckpt:
                ckpt = CheckpointType.BEST
            elif "last" in ckpt:
                ckpt = CheckpointType.LAST
            else:
                step = int(ckpt)
                ckpt = CheckpointType.OTHER
        elif isinstance(ckpt, int):
            step = ckpt
            ckpt = CheckpointType.OTHER

        if ckpt is CheckpointType.BEST:
            ckpt_path = get_best_checkpoint(self._config.model_dir)
            step = int(ckpt_path.name[11:])
        elif ckpt is CheckpointType.LAST:
            ckpt_path = Path(get_last_checkpoint(self._config.model_dir))
            step = int(ckpt_path.name[11:])
        elif ckpt is CheckpointType.OTHER and step is not None:
            ckpt_path = self._config.model_dir / f"checkpoint-{step}"
        else:
            raise ValueError(f"Unsupported checkpoint type: {ckpt}.")
        return ckpt_path, step

    def _create_training_arguments(self) -> Seq2SeqTrainingArguments:
        parser = HfArgumentParser(Seq2SeqTrainingArguments)
        args: dict = {}
        for section, params in TRAINING_ARGS_CONFIG_MAPPING.items():
            section_config: dict = self._config.root[section]
            for param in params:
                if param in section_config:
                    args[param] = section_config[param]
        # For context on floating point precision, see https://github.com/sillsdev/silnlp/issues/647
        merge_dict(
            args,
            {
                "fp16": self._mixed_precision and not self._is_t5,
                "bf16": self._mixed_precision and self._is_t5,
                "tf32": self._mixed_precision,
            },
        )
        if self._config.train["use_lora"] and "learning_rate" not in args.keys():
            args["learning_rate"] = 3e-4
        return parser.parse_dict(args)[0]

    def _get_dictionary(self) -> Dict[VerseRef, Set[str]]:
        if self._dictionary is not None:
            return self._dictionary

        self._dictionary = {}

        dict_trg_path = self._config.exp_dir / self._config.dict_trg_filename()
        dict_vref_path = self._config.exp_dir / self._config.dict_vref_filename()

        if not dict_trg_path.is_file() or not dict_vref_path.is_file():
            return self._dictionary

        with (
            dict_trg_path.open("r", encoding="utf-8-sig") as trg_file,
            dict_vref_path.open("r", encoding="utf-8-sig") as dict_file,
        ):
            for trg_line, vref_line in zip(trg_file, dict_file):
                vref_line = vref_line.strip()
                if vref_line == "":
                    continue
                vref_strs = vref_line.split("\t")
                for vref_str in vref_strs:
                    verse_ref = VerseRef.from_string(vref_str, ORIGINAL_VERSIFICATION)
                    terms = self._dictionary.get(verse_ref)
                    if terms is None:
                        terms = set()
                        self._dictionary[verse_ref] = terms
                    terms.add(trg_line.strip())

        return self._dictionary

    # Untie full embedding modules and instead tie embedding weights
    def _create_tied_embedding_weights(self, model: PreTrainedModel) -> PreTrainedModel:
        encoder_embeddings = torch.nn.Embedding(
            model.config.vocab_size, model.config.d_model, model.config.pad_token_id
        )
        decoder_embeddings = torch.nn.Embedding(
            model.config.vocab_size, model.config.d_model, model.config.pad_token_id
        )

        if self._config.model_prefix == "facebook/nllb-200":
            model.model.encoder.embed_tokens = encoder_embeddings
            model.model.decoder.embed_tokens = decoder_embeddings
            model.tie_weights()
        elif self._config.model_prefix == "google/madlad400":
            model.encoder.embed_tokens = encoder_embeddings
            model.decoder.embed_tokens = decoder_embeddings
            model._tie_or_clone_weights(model.encoder.embed_tokens, model.shared)
            model._tie_or_clone_weights(model.decoder.embed_tokens, model.shared)

        return model

    def _convert_to_lora_model(self, model: PreTrainedModel) -> PreTrainedModel:
        lora_config = self._config.train["lora_config"]
        target_modules = lora_config.get(
            "target_modules", LORA_DEFAULT_CONFIGS[self._config.model_prefix]["target_modules"]
        )
        modules_to_save = lora_config.get(
            "modules_to_save", LORA_DEFAULT_CONFIGS[self._config.model_prefix]["modules_to_save"]
        )
        if isinstance(target_modules, str):
            target_modules = target_modules.split(",")
        if isinstance(modules_to_save, str):
            modules_to_save = modules_to_save.split(",")

        # Only tie embedding weights together rather than the entire modules so that peft recognizes each one
        if "embed_tokens" in modules_to_save or "embed_tokens" in target_modules:
            model = self._create_tied_embedding_weights(model)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_config.get("r", 4),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.1),
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, peft_config)

        if self._config.model_prefix == "facebook/nllb-200" and (
            ("embed_tokens" in modules_to_save and "lm_head" not in modules_to_save)
            or ("lm_head" in modules_to_save and "embed_tokens" not in modules_to_save)
        ):
            LOGGER.warning(
                "NLLB is typically trained with the embeddings tied. "
                "Add both embed_tokens and lm_head to modules_to_save to do this while using LoRA."
            )

        # Tie LoRA copies of the embedding weights together
        if "embed_tokens" in modules_to_save:
            if self._config.model_prefix == "facebook/nllb-200":
                embedding = model.base_model.model.model.encoder.embed_tokens.modules_to_save.default.weight
                model.base_model.model.model.decoder.embed_tokens.modules_to_save.default.weight = embedding
                if "lm_head" in modules_to_save:
                    model.base_model.model.lm_head.modules_to_save.default.weight = embedding
            elif self._config.model_prefix == "google/madlad400":
                embedding = model.base_model.model.encoder.embed_tokens.modules_to_save.default.weight
                model.base_model.model.decoder.embed_tokens.modules_to_save.default.weight = embedding
        elif "embed_tokens" in target_modules:
            if self._config.model_prefix == "facebook/nllb-200":
                # TODO: figure out how to tie embedding weights and lm_head weights together
                embedding_A = model.base_model.model.model.encoder.embed_tokens.lora_embedding_A.default
                embedding_B = model.base_model.model.model.encoder.embed_tokens.lora_embedding_B.default
                model.base_model.model.model.decoder.embed_tokens.lora_embedding_A.default = embedding_A
                model.base_model.model.model.decoder.embed_tokens.lora_embedding_B.default = embedding_B
            elif self._config.model_prefix == "google/madlad400":
                embedding_A = model.base_model.model.encoder.embed_tokens.lora_embedding_A.default
                embedding_B = model.base_model.model.encoder.embed_tokens.lora_embedding_B.default
                model.base_model.model.decoder.embed_tokens.lora_embedding_A.default = embedding_A
                model.base_model.model.decoder.embed_tokens.lora_embedding_B.default = embedding_B

        # Necessary to allow gradients to propogate through frozen layers
        # when using PEFT + gradient checkpointing + Trainer
        if self._config.train["gradient_checkpointing"]:
            model.enable_input_require_grads()

        return model

    def _merge_and_delete_adapter(self, checkpoint_path: Path, vocab_size: int, save_safetensors: bool) -> None:
        adapter_path = checkpoint_path / "adapter"

        base = AutoModelForSeq2SeqLM.from_pretrained(self._config.model)
        base.resize_token_embeddings(vocab_size, pad_to_multiple_of=8 if self._mixed_precision else None)
        base = self._create_tied_embedding_weights(base)

        model_to_merge = PeftModel.from_pretrained(base, adapter_path)
        merged_model = model_to_merge.merge_and_unload()

        if self._config.model_prefix == "facebook/nllb-200":
            embedding_weights = merged_model.model.encoder.embed_tokens.weight
            merged_model.model.shared.weight = embedding_weights
            merged_model.model.decoder.embed_tokens.weight = embedding_weights
            merged_model.lm_head.weight = embedding_weights

        merged_model.save_pretrained(
            checkpoint_path,
            safe_serialization=save_safetensors,
        )

        shutil.rmtree(adapter_path)

    def _translate_sentences(
        self,
        tokenizer: PreTrainedTokenizer,
        pipeline: TranslationPipeline,
        sentences: Iterable[TSent],
        vrefs: Optional[Iterable[VerseRef]],
        produce_multiple_translations: bool = False,
        return_tensors: bool = False,
    ) -> Iterable[OutputGroup]:
        batch_size: int = self._config.infer["infer_batch_size"]

        dictionary = self._get_dictionary()
        if vrefs is None or len(dictionary) == 0:
            yield from self._translate_sentence_helper(
                pipeline,
                sentences,
                batch_size,
                return_tensors,
                produce_multiple_translations=produce_multiple_translations,
            )
        else:
            for batch, force_words in batch_sentences(sentences, vrefs, batch_size, dictionary):
                if force_words is None:
                    force_words_ids = None
                else:
                    force_words_ids = [[tokenizer.convert_tokens_to_ids(v) for v in vs] for vs in force_words]
                    force_words_ids = prune_sublists(force_words_ids)

                yield from self._translate_sentence_helper(
                    pipeline,
                    batch,
                    batch_size,
                    return_tensors,
                    force_words_ids,
                    produce_multiple_translations=produce_multiple_translations,
                )

    def _translate_sentence_helper(
        self,
        pipeline: TranslationPipeline,
        sentences: Iterable[TSent],
        batch_size: int,
        return_tensors: bool,
        force_words_ids: List[List[List[int]]] = None,
        produce_multiple_translations: bool = False,
    ) -> Iterable[OutputGroup]:

        num_drafts = self.get_num_drafts()
        if produce_multiple_translations and num_drafts > 1:
            multiple_translations_method: str = self._config.infer.get("multiple_translations_method")

            sentences = list(sentences)

            if multiple_translations_method == "hybrid":
                beam_search_results: List[dict] = self._translate_with_beam_search(
                    pipeline,
                    sentences,
                    batch_size,
                    return_tensors,
                    num_return_sequences=1,
                    force_words_ids=force_words_ids,
                )

                sampling_results: List[dict] = self._translate_with_sampling(
                    pipeline,
                    sentences,
                    batch_size,
                    return_tensors,
                    num_return_sequences=num_drafts - 1,
                    force_words_ids=force_words_ids,
                )

                # concatenate the beam search results with the sampling results
                yield from [
                    OutputGroup(beam_search_results[i] + sampling_results[i]) for i in range(len(beam_search_results))
                ]

            elif multiple_translations_method == "sampling":
                yield from [
                    OutputGroup(result)
                    for result in self._translate_with_sampling(
                        pipeline,
                        sentences,
                        batch_size,
                        return_tensors,
                        num_return_sequences=num_drafts,
                        force_words_ids=force_words_ids,
                    )
                ]

            elif multiple_translations_method == "beam_search":
                yield from [
                    OutputGroup(result)
                    for result in self._translate_with_beam_search(
                        pipeline,
                        sentences,
                        batch_size,
                        return_tensors,
                        num_return_sequences=num_drafts,
                        force_words_ids=force_words_ids,
                    )
                ]

            elif multiple_translations_method == "diverse_beam_search":
                yield from [
                    OutputGroup(result)
                    for result in self._translate_with_diverse_beam_search(
                        pipeline,
                        sentences,
                        batch_size,
                        return_tensors,
                        num_return_sequences=num_drafts,
                        force_words_ids=force_words_ids,
                    )
                ]
            else:
                LOGGER.error('Unrecognized value for multiple_translations_method: "%s"', multiple_translations_method)

        else:
            yield from [
                OutputGroup([translated_sentence[0]])
                for translated_sentence in self._translate_with_beam_search(
                    pipeline,
                    sentences,
                    batch_size,
                    return_tensors,
                    num_return_sequences=1,
                    force_words_ids=force_words_ids,
                )
            ]

    # When translating tokenized sentences, for some reason the Huggingface pipeline
    # returns List[List[dict]] instead of List[dict]. Each nested list is a
    # singleton. This function flattens the structure.
    def _flatten_tokenized_translations(self, pipeline_output) -> List[dict]:
        return [[i if isinstance(i, dict) else i[0] for i in translation] for translation in pipeline_output]

    def _translate_with_beam_search(
        self,
        pipeline: TranslationPipeline,
        sentences: Iterable[TSent],
        batch_size: int,
        return_tensors: bool,
        num_return_sequences: int = 1,
        force_words_ids: List[List[List[int]]] = None,
    ) -> List[List[dict]]:
        num_beams: Optional[int] = self._config.infer.get("num_beams")
        if num_beams is None:
            num_beams = self._config.params.get("generation_num_beams")

        translations = pipeline(
            sentences,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            force_words_ids=force_words_ids,
            batch_size=batch_size,
            return_text=not return_tensors,
            return_tensors=return_tensors,
        )

        if num_return_sequences == 1:
            translations = [[t] for t in translations]

        return self._flatten_tokenized_translations(translations)

    def _translate_with_sampling(
        self,
        pipeline: TranslationPipeline,
        sentences: Iterable[TSent],
        batch_size: int,
        return_tensors: bool,
        num_return_sequences: int = 1,
        force_words_ids: List[List[List[int]]] = None,
    ) -> List[List[dict]]:

        temperature: Optional[int] = self._config.infer.get("temperature")

        translations = pipeline(
            sentences,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            force_words_ids=force_words_ids,
            batch_size=batch_size,
            return_text=not return_tensors,
            return_tensors=return_tensors,
        )

        if num_return_sequences == 1:
            translations = [[t] for t in translations]

        return self._flatten_tokenized_translations(translations)

    def _translate_with_diverse_beam_search(
        self,
        pipeline: TranslationPipeline,
        sentences: Iterable[TSent],
        batch_size: int,
        return_tensors: bool,
        num_return_sequences: int = 1,
        force_words_ids: List[List[List[int]]] = None,
    ) -> List[List[dict]]:
        num_beams: Optional[int] = self._config.infer.get("num_beams")
        if num_beams is None:
            num_beams = self._config.params.get("generation_num_beams")
        diversity_penalty: Optional[float] = self._config.infer.get("diversity_penalty")

        translations = pipeline(
            sentences,
            num_beams=num_beams,
            num_beam_groups=num_beams,
            num_return_sequences=num_return_sequences,
            diversity_penalty=diversity_penalty,
            force_words_ids=force_words_ids,
            batch_size=batch_size,
            return_text=not return_tensors,
            return_tensors=return_tensors,
        )

        if num_return_sequences == 1:
            translations = [[t] for t in translations]

        return self._flatten_tokenized_translations(translations)

    def _create_inference_model(
        self,
        ckpt: Union[CheckpointType, str, int],
        tokenizer: PreTrainedTokenizer,
        src_lang: str,
        trg_lang: str,
    ) -> PreTrainedModel:
        if self._config.model_dir.exists():
            checkpoint_path, _ = self.get_checkpoint_path(ckpt)
            model_name = str(checkpoint_path)
        else:
            LOGGER.warning("Model has no checkpoints. Using base model.")
            model_name = self._config.model

        dtype = torch.bfloat16 if self._is_t5 else torch.float16
        if (
            self._config.train["use_lora"]
            and self._config.model_prefix != "facebook/nllb-200"
            and model_name != self._config.model
        ):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self._config.model,
                torch_dtype=dtype if self._mixed_precision else "auto",
                attn_implementation=self._config.params["attn_implementation"],
            )
            if len(tokenizer) != base_model.get_input_embeddings().weight.size(dim=0):
                base_model.resize_token_embeddings(
                    len(tokenizer), pad_to_multiple_of=8 if self._mixed_precision else None
                )
            base_model = self._create_tied_embedding_weights(base_model)
            model = PeftModel.from_pretrained(base_model, model_name)
        else:
            model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=dtype if self._mixed_precision else "auto",
                attn_implementation=self._config.params["attn_implementation"],
            )
        if self._config.infer.get("better_transformer"):
            model = model.to_bettertransformer()
        if model_name == self._config.model and len(tokenizer) != model.get_input_embeddings().weight.size(dim=0):
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8 if self._mixed_precision else None)
        if self._config.model_prefix == "google/madlad400" or model_name == self._config.model:
            model, tokenizer = self._configure_model(model, tokenizer, src_lang, trg_lang)

        return model

    def _configure_model(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, src_lang: str, trg_lang: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        # Set decoder_start_token_id
        if (
            trg_lang != ""
            and model.config.decoder_start_token_id is None
            and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast))
        ):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[trg_lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(trg_lang)

        if self._config.model_prefix == "google/madlad400":
            model.config.decoder_start_token_id = tokenizer.pad_token_id
            model.generation_config.decoder_start_token_id = tokenizer.pad_token_id
            model.config.max_length = 256
            model.generation_config.max_new_tokens = 256
            tokenizer.tgt_lang = trg_lang

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
            src_lang != ""
            and trg_lang != ""
            and isinstance(
                tokenizer, (MBartTokenizer, MBartTokenizerFast, M2M100Tokenizer, NllbTokenizer, NllbTokenizerFast)
            )
        ):
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = trg_lang

            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token.
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(trg_lang)
            model.config.forced_bos_token_id = forced_bos_token_id
            if model.generation_config is not None:
                model.generation_config.forced_bos_token_id = forced_bos_token_id

        return model, tokenizer


class PunctuationNormalizingTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self._wrapped_tokenizer = tokenizer
        self._tokenizer = tokenizer._tokenizer
        self._mpn = MosesPunctNormalizer()
        self._mpn.substitutions = [(re.compile(r), sub) for r, sub in self._mpn.substitutions]
        self._pad_token = tokenizer._pad_token

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]] = None,
        text_pair: Union[str, List[str], List[List[str]]] = None,
        text_target: Union[str, List[str], List[List[str]]] = None,
        text_pair_target: Union[str, List[str], List[List[str]]] = None,
        **kwargs,
    ) -> BatchEncoding:
        if text is None:
            raise ValueError('"text" input to PunctuationNormalizingTokenizer cannot be None')

        if isinstance(text, str):
            text = self._mpn.normalize(text)
        elif isinstance(text, (list, tuple)) and len(text) > 0:
            if isinstance(text[0], (list, tuple)) and len(text[0]) > 0:
                text = [[self._mpn.normalize(item) for item in row] for row in text]
            text = [self._mpn.normalize(item) for item in text]
        return self._wrapped_tokenizer(text, **kwargs)

    def token_to_id(self, token: str) -> int:
        return self._wrapped_tokenizer.token_to_id(token)

    def decode(self, *args, **kwargs):
        return self._wrapped_tokenizer.decode(*args, **kwargs)


class HuggingFaceTokenizer(Tokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        lang_codes: Dict[str, str],
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._mpn = MosesPunctNormalizer()
        self._mpn.substitutions = [(re.compile(r), sub) for r, sub in self._mpn.substitutions]
        self._all_special_tokens = set(self._tokenizer.all_special_tokens)
        self._lang_codes = lang_codes
        self._max_source_length = max_source_length
        self._max_target_length = max_target_length

    def set_src_lang(self, src_lang: str) -> None:
        self._tokenizer.src_lang = self._lang_codes.get(src_lang, src_lang)

    def set_trg_lang(self, trg_lang: str) -> None:
        self._tokenizer.tgt_lang = self._lang_codes.get(trg_lang, trg_lang)

    def tokenize(
        self,
        side: Side,
        line: str,
        add_dummy_prefix: bool = True,
        sample_subwords: bool = False,
        add_special_tokens: bool = True,
    ) -> str:
        if isinstance(self._tokenizer, (NllbTokenizer, NllbTokenizerFast)):
            line = self._mpn.normalize(line)
        if not add_dummy_prefix:
            line = "\ufffc" + line
        if side == Side.SOURCE:
            max_length = self._max_source_length
            if isinstance(self._tokenizer, (T5Tokenizer, T5TokenizerFast)):
                line = self._tokenizer.tgt_lang + " " + line
                max_length += 1
            if not add_dummy_prefix:
                max_length += 2
            tokens = self._tokenizer(
                line, add_special_tokens=add_special_tokens, max_length=max_length, truncation=True
            ).tokens()
        else:
            max_length = self._max_target_length
            if not add_dummy_prefix:
                max_length += 2
            tokens = self._tokenizer(
                text_target=line,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=True,
            ).tokens()
        if not add_dummy_prefix:
            tokens.remove("▁")
            tokens.remove("\ufffc")
        return " ".join(t.strip() for t in tokens)

    def normalize_normalized_string(self, line: NormalizedString) -> None:
        if isinstance(self._tokenizer, (NllbTokenizer, NllbTokenizerFast)):
            line.replace(Regex(".+"), self._mpn.normalize(str(line.normalized)))
        self._tokenizer.backend_tokenizer.normalizer.normalize(line)

    def normalize(self, side: Side, line: str) -> str:
        if isinstance(self._tokenizer, (NllbTokenizer, NllbTokenizerFast)):
            line = self._mpn.normalize(line)
        return self._tokenizer.backend_tokenizer.normalizer.normalize_str(line)

    def detokenize(self, line: str) -> str:
        tokens = line.split()
        tokens = [p for p in tokens if p not in self._all_special_tokens]
        return self._tokenizer.clean_up_tokenization(self._tokenizer.convert_tokens_to_string(tokens))


class CustomNormalizerWrapper:
    def __init__(self, tokenizer: HuggingFaceTokenizer) -> None:
        self._tokenizer = tokenizer

    def normalize(self, line: NormalizedString) -> None:
        self._tokenizer.normalize_normalized_string(line)


class SilTranslationPipeline(TranslationPipeline):
    def _forward(self, model_inputs, **generate_kwargs):
        in_b, input_length = model_inputs["input_ids"].shape

        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            config = self.model.generation_config
        else:
            config = self.model.config
        generate_kwargs["min_length"] = generate_kwargs.get("min_length", config.min_length)
        generate_kwargs["max_length"] = generate_kwargs.get("max_length", config.max_length)
        self.check_inputs(input_length, generate_kwargs["min_length"], generate_kwargs["max_length"])
        output = self.model.generate(
            **model_inputs,
            **generate_kwargs,
            output_scores=True,
            return_dict_in_generate=True,
        )

        if isinstance(output, BeamSearchEncoderDecoderOutput):
            output_ids = output.sequences
            beam_indices = output.beam_indices
            scores = output.scores
            sequences_scores = output.sequences_scores
        elif isinstance(output, GreedySearchEncoderDecoderOutput):
            output_ids = output.sequences
            beam_indices = torch.zeros_like(output_ids)
            assert output.scores is not None
            scores = tuple(torch.nn.functional.log_softmax(logits, dim=-1) for logits in output.scores)
            sequences_scores = output.sequences_scores
        else:
            raise RuntimeError("Cannot postprocess the output of the model.")

        assert beam_indices is not None and scores is not None
        out_b = output_ids.shape[0]
        num_beams = scores[0].shape[0] // in_b
        n_sequences = out_b // in_b
        start_index = 0
        if self.model.config.decoder_start_token_id is not None:
            start_index = 1
        indices = torch.stack(
            (
                torch.arange(output_ids.shape[1] - start_index, device=output_ids.device).expand(in_b, n_sequences, -1),
                torch.reshape(beam_indices[:, start_index:] % num_beams, (in_b, n_sequences, -1)),
                torch.reshape(output_ids[:, start_index:], (in_b, n_sequences, -1)),
            ),
            dim=3,
        )
        scores = torch.stack(scores, dim=0).reshape(len(scores), in_b, num_beams, -1).transpose(0, 1)
        scores = torch_gather_nd(scores, indices, 1)
        if self.model.config.decoder_start_token_id is not None:
            scores = torch.cat((torch.zeros(scores.shape[0], scores.shape[1], 1, device=scores.device), scores), dim=2)
        output_ids = output_ids.reshape(in_b, n_sequences, *output_ids.shape[1:])
        return {
            "output_ids": output_ids,
            "scores": scores,
            "sequences_scores": sequences_scores,
        }

    def postprocess(self, model_outputs, return_type=None, clean_up_tokenization_spaces=False):
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer is specified.")

        records = []
        output_ids: torch.Tensor
        scores: torch.Tensor
        for output_ids, scores in zip(
            model_outputs["output_ids"][0],
            model_outputs["scores"][0],
        ):
            output_tokens: List[str] = []
            output_token_ids: List[str] = []
            output_indices: List[int] = []
            for i, output_id in enumerate(output_ids):
                id = cast(int, output_id.item())
                output_tokens.append(self.tokenizer.convert_ids_to_tokens(id))
                output_token_ids.append(id)
                output_indices.append(i)
            scores = scores[output_indices]
            records.append(
                {
                    "translation_tokens": output_tokens,
                    "translation_token_ids": output_token_ids,
                    "token_scores": scores,
                    "sequence_score": model_outputs["sequences_scores"][0],
                    "translation_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    ),
                }
            )
        return records


class PretokenizedTranslationPipeline(SilTranslationPipeline):
    def preprocess(self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None):
        model_inputs = batch_prepare_for_model(self.tokenizer, args, return_tensors=self.framework)
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        model_inputs["forced_bos_token_id"] = tgt_lang_id
        return model_inputs


def torch_gather_nd(params: torch.Tensor, indices: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    """
    torch_gather_nd implements tf.gather_nd in PyTorch.

    This supports multiple batch dimensions as well as multiple channel dimensions.
    """
    index_shape = indices.shape[:-1]
    num_dim = indices.size(-1)
    tail_sizes = params.shape[batch_dim + num_dim :]

    # flatten extra dimensions
    for s in tail_sizes:
        row_indices = torch.arange(s, device=params.device)
        indices = indices.unsqueeze(-2)
        indices = indices.repeat(*[1 for _ in range(indices.dim() - 2)], s, 1)
        row_indices = row_indices.expand(*indices.shape[:-2], -1).unsqueeze(-1)
        indices = torch.cat((indices, row_indices), dim=-1)
        num_dim += 1

    # flatten indices and params to batch specific ones instead of channel specific
    for i in range(num_dim):
        size = prod(params.shape[batch_dim + i + 1 : batch_dim + num_dim])
        indices[..., i] *= size

    indices = indices.sum(dim=-1)
    params = params.flatten(batch_dim, -1)
    indices = indices.flatten(batch_dim, -1)

    out = torch.gather(params, dim=batch_dim, index=indices)
    return out.reshape(*index_shape, *tail_sizes)


class DataCollatorForSeq2SeqNoising:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[Any] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        src_noise: List[NoiseMethod] = [],
        return_tensors: str = "pt",
    ):
        self._data_collator = DataCollatorForSeq2Seq(
            tokenizer, model, padding, max_length, pad_to_multiple_of, label_pad_token_id, return_tensors
        )
        self._src_noise = src_noise

    def __call__(self, features, return_tensors=None):
        if len(self._src_noise) > 0:
            for feature in features:
                input_ids = feature["input_ids"][:-2]
                for noise_method in self._src_noise:
                    input_ids = noise_method(input_ids)
                feature["input_ids"] = input_ids + feature["input_ids"][-2:]
                feature["attention_mask"] = feature["attention_mask"][: len(feature["input_ids"])]

        return self._data_collator(features, return_tensors)


class SilSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[Seq2SeqTrainingArguments] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[optim.Optimizer], Optional[optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        sequential_sampling: bool = False,
        better_transformer: bool = False,
        auto_grad_acc: bool = False,
        model_prefix: Optional[str] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self._sequential_sampling = sequential_sampling
        self._better_transformer = better_transformer
        self._auto_grac_acc = auto_grad_acc
        self.model_prefix = model_prefix

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self._sequential_sampling:
            return None
        return super()._get_train_sampler()

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        if self._auto_grac_acc:
            inner_training_loop = find_executable_batch_size(super()._inner_training_loop, batch_size, self.accelerator)
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        else:
            return super()._inner_training_loop(
                batch_size=batch_size,
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        LOGGER.info(f"Saving model checkpoint to {output_dir} using custom _save function")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                LOGGER.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        elif isinstance(self.model, PeftModel):
            if self._better_transformer:
                self.model = self.model.reverse_bettertransformer()
            if self.model_prefix:
                output_dir += "/adapter" if self.model_prefix == "facebook/nllb-200" else ""
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
                save_embedding_layers=False,
            )
            if self._better_transformer:
                self.model = self.model.to_bettertransformer()
        else:
            if self._better_transformer:
                self.model = self.model.reverse_bettertransformer()
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )
            if self._better_transformer:
                self.model = self.model.to_bettertransformer()
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 64, accelerator=None):
    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()

        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    accelerator.gradient_accumulation_steps = accelerator.gradient_accumulation_steps * 2
                    kwargs["args"].gradient_accumulation_steps = accelerator.gradient_accumulation_steps
                else:
                    raise

    return decorator
