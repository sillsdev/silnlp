import json
import logging
import os
import re
from contextlib import ExitStack
from copy import deepcopy
from enum import Enum
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, TextIO, Tuple, TypeVar, Union, cast

import datasets.utils.logging as datasets_logging
import evaluate
import numpy as np
import torch
import transformers.utils.logging as transformers_logging
import yaml
from datasets import Dataset
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef
from sacremoses import MosesPunctNormalizer
from tokenizers import AddedToken, NormalizedString, Regex, SentencePieceBPETokenizer
from tokenizers.normalizers import Normalizer
from torch import Tensor, TensorType, nn, optim
from torch.utils.checkpoint import checkpoint  # noqa: 401
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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TranslationPipeline,
    set_seed,
)
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
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

from ..common.corpus import count_lines, get_terms
from ..common.environment import SIL_NLP_ENV, download_if_s3_paths
from ..common.utils import NoiseMethod, ReplaceRandomToken, Side, create_noise_methods, merge_dict
from .config import CheckpointType, Config, CorpusPair, NMTModel
from .tokenizer import NullTokenizer, Tokenizer

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel

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
        "evaluation_strategy",
        "greater_is_better",
        "include_inputs_for_metrics",
        "load_best_model_at_end",
        "metric_for_best_model",
        "per_device_eval_batch_size",
        "predict_with_generate",
    },
    "params": {
        "adafactor",
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


OPTIMIZER_STATE_FILES = {"optimizer.pt", "rng_state.pth", "scaler.pt", "scheduler.pt"}


def delete_optimizer_state(checkpoint_path: Path) -> None:
    for file in OPTIMIZER_STATE_FILES:
        path = checkpoint_path / file
        if path.is_file():
            path.unlink()


TOKENIZER_FILES = {
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
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
    tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.additional_special_tokens + [lang_code]})
    lang_id = tokenizer.convert_tokens_to_ids(lang_code)
    tokenizer.lang_code_to_id[lang_code] = lang_id
    if isinstance(tokenizer, (NllbTokenizer, MBart50Tokenizer, MBartTokenizer)):
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


class HuggingFaceConfig(Config):
    def __init__(self, exp_dir: Path, config: dict) -> None:
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
                    "save_steps": 1000,
                    "per_device_train_batch_size": 16,
                    "save_strategy": "steps",
                    "save_total_limit": 2,
                    "gradient_accumulation_steps": 4,
                    "max_steps": 100000,
                    "group_by_length": True,
                    "output_dir": str(exp_dir / "run"),
                    "delete_checkpoint_optimizer_state": True,
                    "delete_checkpoint_tokenizer": True,
                    "log_level": "info",
                },
                "eval": {
                    "evaluation_strategy": "steps",
                    "eval_steps": 1000,
                    "early_stopping": {"min_improvement": 0.2, "steps": 4},
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "bleu",
                    "per_device_eval_batch_size": 16,
                    "multi_ref_eval": False,
                    "predict_with_generate": True,
                    "detokenize": True,
                },
                "infer": {"infer_batch_size": 16, "num_beams": 2},
                "params": {
                    "optim": "adamw_torch",
                    "label_smoothing_factor": 0.2,
                    "warmup_steps": 4000,
                    "dropout": 0.1,
                    "attention_dropout": 0.1,
                    "activation_dropout": 0.0,
                },
            },
            config,
        )
        self._tokenizer: Optional[PreTrainedTokenizer] = None

        super().__init__(exp_dir, config)

        # disable evaluation if there is no validation split
        if not self.has_val_split:
            config["eval"]["evaluation_strategy"] = "no"
            config["eval"]["load_best_model_at_end"] = False
            config["eval"]["early_stopping"] = None
            config["eval"]["metric_for_best_model"] = None

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
        self, missing_tokens: List[str], trained_tokenizers: Optional[List[SentencePieceBPETokenizer]] = None
    ) -> None:
        assert self._tokenizer is not None
        self._tokenizer.save_pretrained(str(self.exp_dir))
        with open(self.exp_dir / "tokenizer.json", "r+", encoding="utf-8") as file:
            data = json.load(file)
            vocab_len = len(data["model"]["vocab"].keys())
            for i, token in enumerate(missing_tokens):
                data["model"]["vocab"][token] = vocab_len + i
            if trained_tokenizers:
                for trained_tok in trained_tokenizers:
                    trained_tok.save(str(self.exp_dir / "tokenizer_trained.json"))
                    with open(self.exp_dir / "tokenizer_trained.json", "r+", encoding="utf-8") as trained_file:
                        trained_data = json.load(trained_file)
                        data["model"]["merges"] = trained_data["model"]["merges"] + data["model"]["merges"]
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.truncate()
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.exp_dir), use_fast=True)
        return

    def _train_sp_tokenizer(self, files, vocab_size) -> SentencePieceBPETokenizer:
        assert self._tokenizer is not None
        sp_tok = SentencePieceBPETokenizer()
        hf_tokenizer = HuggingFaceTokenizer(
            self._tokenizer, self.data["lang_codes"], self.train["max_source_length"], self.train["max_target_length"]
        )
        sp_tok.normalizer = Normalizer.custom(CustomNormalizerWrapper(hf_tokenizer))
        sp_tok.train(
            files, vocab_size=vocab_size, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )
        sp_tok.normalizer = self._tokenizer.backend_tokenizer.normalizer
        return sp_tok

    def _create_trained_tokens(self, file_paths, vocab_size) -> Tuple[List[str], SentencePieceBPETokenizer]:
        assert self._tokenizer is not None
        files = [str(f) for f in download_if_s3_paths(file_paths)]
        sp_tokenizer = self._train_sp_tokenizer(files, vocab_size)
        sp_keys, tok_keys = sp_tokenizer.get_vocab().keys(), self._tokenizer.get_vocab().keys()
        missing_tokens = sorted(list(set(sp_keys) - set(tok_keys)))
        return missing_tokens, sp_tokenizer

    def _find_missing_characters(self, corpus: List[Path]) -> List[str]:
        assert self._tokenizer is not None
        vocab = self._tokenizer.get_vocab().keys()
        charset: Set[str] = set()
        for file in corpus:
            with file.open("r", encoding="utf-8-sig") as f:
                charset = charset | set(f.read())
        hf_tokenizer = HuggingFaceTokenizer(
            self._tokenizer, self.data["lang_codes"], self.train["max_source_length"], self.train["max_target_length"]
        )
        charset = set(hf_tokenizer.normalize_all(Side.TARGET, charset))
        charset = set(filter(None, {char.strip() for char in charset}))
        missing_characters = sorted(list(charset - vocab))
        return missing_characters

    def _build_vocabs(self, stats: bool = False) -> None:
        tok_dict = self.data.get("tokenizer")
        self._tokenizer = self.get_or_create_tokenizer()
        tokens: List[str] = []
        trained_tokenizers = []
        if tok_dict and (tok_dict.get("update_src") or tok_dict.get("update_trg")):
            if tok_dict.get("trained_tokens") and (SIL_NLP_ENV.assets_dir / "tokenizer_config.json").is_file():
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
                    elif tok_dict.get("update_trg"):
                        missing_tokens, trained_tokenizer = self._create_trained_tokens(
                            list(self.trg_file_paths), tok_dict.get("trg_vocab_size")
                        )
                    trained_tokenizers.append(trained_tokenizer)
            else:
                if tok_dict.get("update_src") and tok_dict.get("update_trg"):
                    file_paths = list(self.src_file_paths) + list(self.trg_file_paths)
                elif tok_dict.get("update_src"):
                    file_paths = list(self.src_file_paths)
                elif tok_dict.get("update_trg"):
                    file_paths = list(self.trg_file_paths)
                missing_tokens = self._find_missing_characters(file_paths)
            if stats:
                with ExitStack() as stack:
                    stats_file: Optional[TextIO] = None
                    stats_file = stack.enter_context(
                        (self.exp_dir / "tokenization_stats.txt").open("w", encoding="utf-8", newline="\n")
                    )
                    stats_file.write(f"Added tokens: {len(missing_tokens)}\n")
            tokens += missing_tokens
        if tokens:
            self._add_tokens(tokens, trained_tokenizers)

        if self.data["add_new_lang_code"]:
            lang_codes: Dict[str, str] = self.data["lang_codes"]
            updated = False
            for iso in self.src_isos | self.trg_isos:
                lang_code = lang_codes.get(iso, iso)
                if lang_code not in self._tokenizer.lang_code_to_id:
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
                and (self.exp_dir / "sentencepiece.bpe.model").is_file()
                and not (self.exp_dir / "tokenizer_config.json").is_file()
            ):
                self._tokenizer = NllbTokenizer.from_pretrained(str(self.exp_dir))
                self._tokenizer = convert_slow_tokenizer(self._tokenizer)
                self._tokenizer = NllbTokenizerFast(tokenizer_object=self._tokenizer)
                self._tokenizer.save_pretrained(str(self.exp_dir))
            else:
                if (not tok_dict or not (tok_dict.get("update_src") or tok_dict.get("update_trg"))) and (
                    self.exp_dir / "tokenizer_config.json"
                ).is_file():
                    model_name_or_path = str(self.exp_dir)
                elif (tok_dict and (tok_dict.get("update_src") or tok_dict.get("update_trg"))) and (
                    SIL_NLP_ENV.assets_dir / "tokenizer_config.json"
                ).is_file():
                    model_name_or_path = str(SIL_NLP_ENV.assets_dir)
                else:
                    model_name_or_path = self.model
                self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            self._tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return self._tokenizer

    def get_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            model_name_or_path = str(self.exp_dir) if (self.exp_dir / "tokenizer_config.json").is_file() else self.model
            self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            self._tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return self._tokenizer

    def _write_dictionary(self, tokenizer: Tokenizer, pair: CorpusPair) -> int:
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

            all_trg_terms = [
                (trg_terms_file, get_terms(trg_terms_file.path)) for trg_terms_file in pair.trg_terms_files
            ]
            for trg_terms_file, trg_terms in all_trg_terms:
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

            if terms_config["include_glosses"] and "en" in self.trg_isos:
                all_src_terms = [get_terms(src_terms_file.path) for src_terms_file in pair.src_terms_files]
                tokenizer.set_trg_lang("en")
                for src_terms in all_src_terms:
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


class HuggingFaceNMTModel(NMTModel):
    def __init__(self, config: HuggingFaceConfig, mixed_precision: bool, num_devices: int) -> None:
        self._config = config
        self._mixed_precision = mixed_precision
        set_seed(self._config.data["seed"])
        self._dictionary: Optional[Dict[VerseRef, Set[str]]] = None

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
        )
        model = cast(PreTrainedModel, AutoModelForSeq2SeqLM.from_pretrained(self._config.model, config=model_config))
        if self._config.train.get("better_transformer"):
            model = model.to_bettertransformer()
        tokenizer = self._config.get_tokenizer()

        old_embeddings = model.get_input_embeddings()
        old_num_tokens = old_embeddings.weight.size(dim=0)
        tok_dict = self._config.data.get("tokenizer")
        if len(tokenizer) > old_num_tokens and tok_dict is not None and tok_dict.get("init_unk"):
            vocab = tokenizer.get_vocab()
            unk_embedding = old_embeddings.weight.data[vocab["<unk>"]]
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8 if training_args.fp16 else None)
            embeddings = model.get_input_embeddings()
            embeddings.weight.data[old_num_tokens:, :] = unk_embedding
            model.tie_weights()
        elif len(tokenizer) != old_num_tokens:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8 if training_args.fp16 else None)

        # Set decoder_start_token_id
        if (
            self._config.val_trg_lang != ""
            and model.config.decoder_start_token_id is None
            and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast))
        ):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[self._config.val_trg_lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(self._config.val_trg_lang)

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if self._config.val_src_lang != "" and self._config.val_trg_lang != "":
            tokenizer.src_lang = self._config.val_src_lang
            tokenizer.tgt_lang = self._config.val_trg_lang

            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token.
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(self._config.val_trg_lang)
            model.config.forced_bos_token_id = forced_bos_token_id
            if model.generation_config is not None:
                model.generation_config.forced_bos_token_id = forced_bos_token_id

        def load_text_dataset(src_path: Path, trg_path: Path) -> Optional[Dataset]:
            if not src_path.is_file() or not trg_path.is_file():
                return None
            data = []
            with open(src_path, "r", encoding="utf-8-sig") as src_file, open(
                trg_path, "r", encoding="utf-8-sig"
            ) as trg_file:
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
            pad_to_multiple_of=8 if training_args.fp16 else None,
            src_noise=src_noise,
        )

        metric = evaluate.load("sacrebleu")
        all_special_ids = set(tokenizer.all_special_ids)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]

            # Replace -100 in the labels as we can't decode them.
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

            result = metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                lowercase=True,
                force=not self._config.eval["detokenize"],
            )
            result = {"bleu": result["score"]}

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
            compute_metrics=compute_metrics,
            sequential_sampling=self._config.train.get("sequential_sampling", False),
            better_transformer=self._config.train.get("better_transformer", False),
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
        if delete_checkpoint_optimizer_state or delete_checkpoint_tokenizer:
            for child in Path(training_args.output_dir).iterdir():
                if child.is_dir() and child.name.startswith("checkpoint-"):
                    if delete_checkpoint_optimizer_state:
                        delete_optimizer_state(child)
                    if delete_checkpoint_tokenizer:
                        delete_tokenizer(child)

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
        vref_paths: Optional[List[Path]] = None,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        checkpoint_path, _ = self.get_checkpoint_path(ckpt)
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_path), torch_dtype=torch.float16 if self._mixed_precision else "auto")
        if self._config.infer.get("better_transformer"):
            model = model.to_bettertransformer()
        tokenizer = self._config.get_tokenizer()
        pipeline = PretokenizedTranslationPipeline(
            model=model,
            tokenizer=tokenizer,
            src_lang=self._config.test_src_lang,
            tgt_lang=self._config.test_trg_lang,
            device=0,
        )
        for input_path, translation_path, vref_path in zip(
            input_paths,
            translation_paths,
            cast(Iterable[Optional[Path]], repeat(None) if vref_paths is None else vref_paths),
        ):
            length = count_lines(input_path)
            with ExitStack() as stack:
                src_file = stack.enter_context(input_path.open("r", encoding="utf-8-sig"))
                sentences = (line.strip().split() for line in src_file)
                out_file = stack.enter_context(translation_path.open("w", encoding="utf-8", newline="\n"))
                vrefs: Optional[Iterable[VerseRef]] = None
                if vref_path is not None:
                    vref_file = stack.enter_context(vref_path.open("r", encoding="utf-8-sig"))
                    vrefs = (VerseRef.from_string(line.strip(), ORIGINAL_VERSIFICATION) for line in vref_file)

                for prediction in tqdm(
                    self._translate_sentences(tokenizer, pipeline, sentences, vrefs, return_tensors=True),
                    total=length,
                    unit="ex",
                ):
                    ids = to_py_obj(prediction[0]["translation_token_ids"])
                    ids = [id for id in ids[1:] if id != tokenizer.pad_token_id]
                    tokens = tokenizer.convert_ids_to_tokens(ids)
                    out_file.write(" ".join(tokens) + "\n")

    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        vrefs: Optional[Iterable[VerseRef]] = None,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Iterable[str]:
        if self._config.model_dir.exists():
            checkpoint_path, _ = self.get_checkpoint_path(ckpt)
            model_name = str(checkpoint_path)
        else:
            model_name = self._config.model
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16 if self._mixed_precision else "auto")
        if self._config.infer.get("better_transformer"):
            model = model.to_bettertransformer()
        if model.config.max_length < 512:
            model.config.max_length = 512
        tokenizer = self._config.get_tokenizer()
        lang_codes: Dict[str, str] = self._config.data["lang_codes"]
        pipeline = TranslationPipeline(
            model=model,
            tokenizer=tokenizer,
            src_lang=lang_codes.get(src_iso, src_iso),
            tgt_lang=lang_codes.get(trg_iso, trg_iso),
            device=0,
        )
        if not isinstance(sentences, list):
            sentences = list(sentences)
        for outputs in tqdm(
            self._translate_sentences(tokenizer, pipeline, sentences, vrefs),
            total=len(sentences),
            unit="ex",
        ):
            if isinstance(outputs, dict):
                outputs = [outputs]
            yield from (p["translation_text"] for p in outputs)

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
        merge_dict(args, {"fp16": self._mixed_precision, "tf32": self._mixed_precision})
        return parser.parse_dict(args)[0]

    def _get_dictionary(self) -> Dict[VerseRef, Set[str]]:
        if self._dictionary is not None:
            return self._dictionary

        self._dictionary = {}

        dict_trg_path = self._config.exp_dir / self._config.dict_trg_filename()
        dict_vref_path = self._config.exp_dir / self._config.dict_vref_filename()

        if not dict_trg_path.is_file() or not dict_vref_path.is_file():
            return self._dictionary

        with dict_trg_path.open("r", encoding="utf-8-sig") as trg_file, dict_vref_path.open(
            "r", encoding="utf-8-sig"
        ) as dict_file:
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

    def _translate_sentences(
        self,
        tokenizer: PreTrainedTokenizer,
        pipeline: TranslationPipeline,
        sentences: Iterable[TSent],
        vrefs: Optional[Iterable[VerseRef]],
        return_tensors: bool = False,
    ) -> Iterable[dict]:
        batch_size: int = self._config.infer["infer_batch_size"]
        num_beams: Optional[int] = self._config.infer.get("num_beams")
        if num_beams is None:
            num_beams = self._config.params.get("generation_num_beams")

        dictionary = self._get_dictionary()
        if vrefs is None or len(dictionary) == 0:
            yield from pipeline(
                sentences,
                num_beams=num_beams,
                batch_size=batch_size,
                return_text=not return_tensors,
                return_tensors=return_tensors,
            )
        else:
            for batch, force_words in batch_sentences(sentences, vrefs, batch_size, dictionary):
                if force_words is None:
                    force_words_ids = None
                else:
                    force_words_ids = [[tokenizer.convert_tokens_to_ids(v) for v in vs] for vs in force_words]
                    force_words_ids = prune_sublists(force_words_ids)

                yield from pipeline(
                    batch,
                    num_beams=num_beams,
                    force_words_ids=force_words_ids,
                    batch_size=batch_size,
                    return_text=not return_tensors,
                    return_tensors=return_tensors,
                )


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
        line = self._mpn.normalize(line)
        if not add_dummy_prefix:
            line = "\ufffc" + line
        if side == Side.SOURCE:
            max_length = self._max_source_length
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
        line.replace(Regex(".+"), self._mpn.normalize(str(line.normalized)))
        self._tokenizer.backend_tokenizer.normalizer.normalize(line)

    def normalize(self, side: Side, line: str) -> str:
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


class PretokenizedTranslationPipeline(TranslationPipeline):
    def preprocess(self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None):
        model_inputs = batch_prepare_for_model(self.tokenizer, args, return_tensors=self.framework)
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        model_inputs["forced_bos_token_id"] = tgt_lang_id
        return model_inputs


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

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self._sequential_sampling:
            return None
        return super()._get_train_sampler()

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
        else:
            if self._better_transformer:
                self.model = self.model.reverse_bettertransformer()
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
                self.model = self.model.to_bettertransformer()
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
