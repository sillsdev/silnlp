import gc
import logging
import os
import re
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple, TypeVar, Union, cast

import datasets.utils.logging as datasets_logging
import evaluate
import numpy as np
import safetensors.torch
import torch
import transformers.utils.logging as transformers_logging
from accelerate.utils.memory import should_reduce_batch_size
from datasets import Dataset
from machine.scripture import VerseRef
from torch import Tensor, nn, optim
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler
from tqdm.std import tqdm as std_tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBartTokenizer,
    NllbTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TensorType,
    TrainerCallback,
    set_seed,
)
from transformers.modeling_utils import unwrap_model
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import SAFE_WEIGHTS_NAME
from transformers.utils.generic import PaddingStrategy, to_py_obj
from transformers.utils.logging import tqdm

from ..common.corpus import count_lines
from ..common.environment import SilNlpEnv
from ..common.translation_data_structures import DraftGroup, SentenceTranslation, SentenceTranslationGroup
from ..common.translator import generate_confidence_files
from ..common.utils import NoiseMethod, ReplaceRandomToken, create_noise_methods, merge_dict
from .checkpoints import CheckpointDirectory, CheckpointType
from .config import (
    Config,
    InferenceModelParams,
    NMTModel,
)
from .training_arguments import TrainingArgumentsMapping
from .config_keys import RenamedConfigKeys
from .dictionary_writer import DictionaryWriter, TermDictionaryWriter
from .decoder_inputs import DecoderInputs
from .huggingface_tokenizer import PunctuationNormalizingTokenizer
from .experiment_settings import TrainingSettings
from .model_name import ModelName
from .parent_model import ParentModel
from .pretrained_tokenizer import PretrainedTokenizer
from .tokenizer_settings import TokenizerSettings, TokenizerSource
from .vocabulary import LanguageCodes, MissingTokens, TokenizerVocabularyBuilder
from .vocabulary_builder import VocabularyBuilder
from .tokenizer import NullTokenizer, Tokenizer

LOGGER = logging.getLogger(__name__)


_TRAINING_ARGS_CONFIG_MAPPING = {
    "train": {
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "log_level",
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
        "train_sampling_strategy",
    },
    "eval": {
        "eval_accumulation_steps",
        "eval_delay",
        "eval_steps",
        "eval_strategy",
        "greater_is_better",
        "include_for_metrics",
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
        "warmup_steps",
        "weight_decay",
    },
}

# Config keys renamed from huggingface 4.x to 5.x, so we need to warn team members if they have them in their config
# rather than silently dropping the arguments. Can be removed once team is accustomed to 5.x.
# "loss" and "eval_loss" are both evaluation loss
# The early stopping callback adds "eval_" to all metrics that don't already start with it
DEFAULT_METRICS = ["loss", "eval_loss"]
EVAL_METRICS_MODULES = {
    "bleu": "sacrebleu",
    "chrf3": "chrf",
    "chrf3+": "chrf",
    "chrf3++": "chrf",
    "m-bleu": "sacrebleu",
    "m-chrf3": "chrf",
    "m-chrf3+": "chrf",
    "m-chrf3++": "chrf",
}


def add_lang_code_to_tokenizer(tokenizer: PreTrainedTokenizerBase, lang_code: str) -> None:
    tokenizer.add_special_tokens({"extra_special_tokens": [lang_code]}, replace_extra_special_tokens=False)
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


class PreTrainedModelProvider(ABC):
    @abstractmethod
    def create_model_for_training(
        self, model_name: str, model_config: Any, device_map: dict[str, int]
    ) -> PreTrainedModel:
        ...

    @abstractmethod
    def create_model_for_inference(self, model_name: str) -> PreTrainedModel:
        ...


class PreTrainedModelProviderFactory(ABC):
    @abstractmethod
    def create_pretrained_model_provider(
        self, config: "Seq2SeqConfig", mixed_precision: bool = False
    ) -> PreTrainedModelProvider:
        ...


class FilePreTrainedModelProvider(PreTrainedModelProvider):
    def __init__(self, attention_implementation: str, dtype: bool | str = "auto") -> None:
        super().__init__()
        self._dtype = dtype
        self._attention_implementation = attention_implementation

    def create_model_for_training(
        self, model_name: str, model_config: Any, device_map: dict[str, int]
    ) -> PreTrainedModel:
        model = cast(
            PreTrainedModel,
            AutoModelForSeq2SeqLM.from_pretrained(model_name, config=model_config, device_map=device_map, token=False),
        )
        return model

    def create_model_for_inference(self, model_name: str) -> PreTrainedModel:
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            attn_implementation=self._attention_implementation,
            token=False,
        )


class FilePreTrainedModelProviderFactory(PreTrainedModelProviderFactory):
    def create_pretrained_model_provider(
        self, config: "Seq2SeqConfig", mixed_precision: bool = False
    ) -> PreTrainedModelProvider:
        attention_implementation = config.params.get("attn_implementation", "sdpa")
        dtype = torch.bfloat16 if config.model_name.is_t5() else torch.float16
        if not mixed_precision:
            dtype = "auto"
        return FilePreTrainedModelProvider(attention_implementation, dtype)


class Seq2SeqConfig(Config):
    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        self.environment = environment
        RenamedConfigKeys.of_seq2seq().warn_about(config)
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
                    "tokenizer": {
                        "init_unk": False,
                        "share_vocab": False,
                        "src_vocab_size": 500,
                        "trained_tokens": False,
                        "trg_vocab_size": 500,
                        "update_src": True,
                        "update_trg": True,
                    },
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
                    "train_sampling_strategy": "group_by_length",
                    "output_dir": str(exp_dir / "run"),
                    "delete_checkpoint_optimizer_state": True,
                    "delete_checkpoint_tokenizer": True,
                    "log_level": "info",
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
        self.model_name = ModelName(config.get("model", ""))

        if "parent" in config["data"]:
            parent = ParentModel(config["data"]["parent"], environment).checkpoint_for(self.model_name)
            config["model"] = parent.path
            self.model_name = parent.family

        super().__init__(exp_dir, config, environment)

        TrainingSettings(self.train).fit_to(self.model_name)
        self._disable_eval_if_no_val_split()

        self._tokenizer_settings = TokenizerSettings(self.data.get("tokenizer"))
        self._tokenizer_source = TokenizerSource(
            self.exp_dir,
            self.model_name.tokenizer_assets_dir(environment.assets_dir),
            environment.get_mt_exp_dir(self.data["parent"]) if self.has_parent else None,
            self.model,
            self._tokenizer_settings,
        )
        self._pretrained_tokenizer = PretrainedTokenizer(
            self._tokenizer_source,
            self.model_name,
            self.exp_dir,
            self.data["lang_codes"],
            self.train["max_source_length"],
            self.train["max_target_length"],
        )

    @property
    def val_src_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.inventory.default_validation_source_iso(), self.inventory.default_validation_source_iso())

    @property
    def test_src_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.inventory.default_test_source_iso(), self.inventory.default_test_source_iso())

    @property
    def val_trg_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.inventory.default_validation_target_iso(), self.inventory.default_validation_target_iso())

    @property
    def test_trg_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.inventory.default_test_target_iso(), self.inventory.default_test_target_iso())

    def create_model(
        self,
        mixed_precision: bool = True,
        num_devices: int = 1,
        clearml_queue: Optional[str] = None,
        pretrained_model_provider_factory: PreTrainedModelProviderFactory = FilePreTrainedModelProviderFactory(),
    ) -> NMTModel:
        return Seq2SeqNMTModel(self, mixed_precision, num_devices, clearml_queue, pretrained_model_provider_factory)

    def create_tokenizer(self) -> Tokenizer:
        if not self.data["tokenize"]:
            return NullTokenizer()
        return self._pretrained_tokenizer.sil_tokenizer()

    def create_vocabulary_builder(self) -> VocabularyBuilder:
        return TokenizerVocabularyBuilder(
            self._pretrained_tokenizer,
            MissingTokens(
                self._pretrained_tokenizer,
                self._tokenizer_source,
                self._tokenizer_settings,
                self.inventory,
                self.model_name,
                self.exp_dir,
            ),
            self.inventory,
            LanguageCodes(self.data["lang_codes"], self.inventory, self.exp_dir),
            self.exp_dir,
            add_new_lang_code=self.data["add_new_lang_code"],
            tokenize=self.data["tokenize"],
        )

    def get_or_create_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._pretrained_tokenizer.build()

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._pretrained_tokenizer.load()

    def _dictionary_writer(self, tokenizer: Tokenizer) -> DictionaryWriter:
        return TermDictionaryWriter(
            self.files, tokenizer, self._term_categories(), self._gloss_language(), self._environment
        )


def batch_prepare_for_model(
    tokenizer: PreTrainedTokenizerBase,
    batch_tokens: List[List[str]],
    return_tensors: Optional[Union[str, TensorType]] = None,
) -> BatchEncoding:
    input_ids = [cast(List[int], tokenizer.convert_tokens_to_ids(tokens)) for tokens in batch_tokens]
    return tokenizer.pad({"input_ids": input_ids}, padding=False, return_tensors=return_tensors)


TSent = TypeVar("TSent")


def batch_sentences(
    sentences: Iterable[TSent],
    batch_size: int,
) -> Iterable[List[TSent]]:
    batch: List[TSent] = []
    for sentence in sentences:
        batch.append(sentence)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


@dataclass
class ModelOutput:
    translated_text: str
    translation_token_ids: List[int]
    token_scores: List[float]
    sequence_score: Optional[float]

    def convert_to_sentence_translation(self, tokenizer: PreTrainedTokenizerBase) -> SentenceTranslation:
        tokens = tokenizer.convert_ids_to_tokens(self.translation_token_ids)
        return SentenceTranslation(
            to_py_obj(self.translated_text),
            to_py_obj(tokens),
            to_py_obj(self.token_scores),
            to_py_obj(self.sequence_score),
        )


# This class represents multiple translations of a single input sequence
class ModelOutputGroup:
    def __init__(self, outputs: List[dict]):
        self._outputs = outputs

    def _get_model_outputs(self) -> List[ModelOutput]:
        return [
            ModelOutput(
                output["translation_text"],
                output["translation_token_ids"],
                output["token_scores"],
                output["sequence_score"],
            )
            for output in self._outputs
        ]

    def convert_to_sentence_translation_group(self, tokenizer: PreTrainedTokenizerBase) -> SentenceTranslationGroup:
        return SentenceTranslationGroup(
            [model_output.convert_to_sentence_translation(tokenizer) for model_output in self._get_model_outputs()]
        )


class Seq2SeqNMTModel(NMTModel):
    def __init__(
        self,
        config: Seq2SeqConfig,
        mixed_precision: bool,
        num_devices: int,
        clearml_queue: Optional[str] = None,
        pretrained_model_provider_factory: PreTrainedModelProviderFactory = FilePreTrainedModelProviderFactory(),
    ) -> None:
        super().__init__(config)
        self._config: Seq2SeqConfig = config
        self._mixed_precision = mixed_precision
        set_seed(self._config.data["seed"])
        self._dictionary: Optional[Dict[VerseRef, Set[str]]] = None
        self._is_t5 = self._config.model_name.is_t5()
        self._num_devices = num_devices
        self._clearml_queue = clearml_queue
        self._pretrained_model_provider = pretrained_model_provider_factory.create_pretrained_model_provider(
            config, mixed_precision
        )

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
            token=False,
        )
        if self._num_devices == 2 and self._config.model_name.is_nllb():
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
        model = self._pretrained_model_provider.create_model_for_training(
            self._config.model, model_config, device_map=device_map
        )

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
        elif len(tokenizer) > old_num_tokens:
            model.resize_token_embeddings(
                len(tokenizer), pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None
            )

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
            self._config.files.train_source(),
            self._config.files.train_target(),
        )

        eval_dataset = load_text_dataset(
            self._config.files.validation_source(),
            self._config.files.validation_target(),
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
            DecoderInputs(model),
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
            processing_class=tokenizer,
            compute_metrics=None if metric_name in DEFAULT_METRICS else compute_metrics,
            sequential_sampling=self._config.train.get("sequential_sampling", False),
            auto_grad_acc=self._config.train.get("auto_grad_acc", False),
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
        written_checkpoints = CheckpointDirectory(Path(training_args.output_dir))
        if delete_checkpoint_optimizer_state:
            written_checkpoints.discard_optimizer_state()
        if delete_checkpoint_tokenizer:
            written_checkpoints.discard_tokenizers()

    def save_effective_config(self, path: Path) -> None:
        TrainingArgumentsMapping(_TRAINING_ARGS_CONFIG_MAPPING).write_effective_config(
            path, self._config.root, self._create_training_arguments()
        )

    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        tokenizer = self._config.get_tokenizer()
        model = self._create_inference_model(ckpt, tokenizer, self._config.test_src_lang, self._config.test_trg_lang)
        compiled_model = cast(PreTrainedModel, torch.compile(model))

        for input_path, translation_path in zip(
            input_paths,
            translation_paths,
        ):
            translator = self._create_translator_for_test_file(input_path, compiled_model, tokenizer)

            length = count_lines(input_path)
            with ExitStack() as stack:
                src_file = stack.enter_context(input_path.open("r", encoding="utf-8-sig"))
                sentences = (line.strip().split() for line in src_file)
                sentence_translation_groups: List[SentenceTranslationGroup] = list(
                    self._translate_test_sentences(
                        tokenizer, translator, sentences, length, produce_multiple_translations
                    )
                )
                draft_group = DraftGroup(sentence_translation_groups)

                for draft_index, translated_draft in enumerate(draft_group.get_drafts(), 1):
                    if produce_multiple_translations:
                        translation_draft_path = translation_path.with_suffix(
                            f".{draft_index}{translation_path.suffix}"
                        )
                    else:
                        translation_draft_path = translation_path
                    out_file = stack.enter_context(translation_draft_path.open("w", encoding="utf-8", newline="\n"))
                    out_file.write("\n".join(translated_draft.get_all_tokenized_translations()) + "\n")

                    if save_confidences:
                        generate_confidence_files(
                            translated_draft,
                            translation_draft_path,
                        )

    def _create_translator_for_test_file(
        self, input_path: Path, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
    ) -> "PretokenizedTranslator":
        iso_specified_file_pattern = re.compile(r"^test\.([a-z]{2,3})\.([a-z]{2,3})\..*")
        if iso_specified_file_pattern.match(input_path.name):
            src_iso, trg_iso = iso_specified_file_pattern.match(input_path.name).groups()
            src_lang = self._config.data["lang_codes"].get(src_iso, src_iso)
            trg_lang = self._config.data["lang_codes"].get(trg_iso, trg_iso)
        else:
            src_lang = self._config.test_src_lang
            trg_lang = self._config.test_trg_lang

        return PretokenizedTranslator(model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=trg_lang)

    def _translate_test_sentences(
        self,
        tokenizer: PreTrainedTokenizerBase,
        translator: "SilTranslator",
        sentences: Iterable[List[str]],
        length: int,
        produce_multiple_translations: bool = False,
    ) -> Iterable[SentenceTranslationGroup]:
        num_drafts = self.get_num_drafts()
        if produce_multiple_translations and num_drafts > 1:
            LOGGER.info("Producing %i translated drafts", num_drafts)
        elif produce_multiple_translations and num_drafts <= 1:
            LOGGER.warning(
                "num_drafts must be greater than 1 when using --multiple-translations. "
                "Falling back to a single translation."
            )

        for model_output_group in tqdm(
            self._translate_sentences(translator, sentences, produce_multiple_translations),
            total=length,
            unit="ex",
        ):
            yield model_output_group.convert_to_sentence_translation_group(tokenizer)

    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Generator[SentenceTranslationGroup, None, None]:
        src_lang = self._config.data["lang_codes"].get(src_iso, src_iso)
        trg_lang = self._config.data["lang_codes"].get(trg_iso, trg_iso)
        inference_model_params = InferenceModelParams(ckpt, src_lang, trg_lang)
        tokenizer = self._config.get_tokenizer()
        if self._inference_model_params == inference_model_params and self._cached_inference_model is not None:
            model = self._cached_inference_model
        else:
            model = self._cached_inference_model = self._create_inference_model(ckpt, tokenizer, src_lang, trg_lang)
            self._inference_model_params = inference_model_params

        # The tokenizer isn't wrapped until after calling _create_inference_model,
        # because the tokenizer's input/output language codes are set there
        if isinstance(tokenizer, NllbTokenizer):
            tokenizer = PunctuationNormalizingTokenizer(tokenizer)

        translator = SilTranslator(
            model=cast(PreTrainedModel, torch.compile(model)),
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=trg_lang,
        )

        num_drafts = self.get_num_drafts()
        if produce_multiple_translations and num_drafts > 1:
            LOGGER.info("Producing %i translated drafts", num_drafts)
        elif produce_multiple_translations and num_drafts <= 1:
            LOGGER.warning(
                "num_drafts must be greater than 1 when using --multiple-translations. "
                "Falling back to a single translation."
            )

        if not isinstance(sentences, list):
            sentences = list(sentences)
        for model_output_group in tqdm(
            self._translate_sentences(translator, sentences, produce_multiple_translations),
            total=len(sentences),
            unit="ex",
        ):
            yield model_output_group.convert_to_sentence_translation_group(tokenizer)

    def _create_training_arguments(self) -> Seq2SeqTrainingArguments:
        args = TrainingArgumentsMapping(_TRAINING_ARGS_CONFIG_MAPPING).collect(
            self._config.root,
            # For context on floating point precision, see https://github.com/sillsdev/silnlp/issues/647
            {
                "fp16": self._mixed_precision and not self._is_t5,
                "bf16": self._mixed_precision and self._is_t5,
                "tf32": self._mixed_precision,
            },
            self._clearml_queue,
        )
        return HfArgumentParser(Seq2SeqTrainingArguments).parse_dict(args)[0]

    def _translate_sentences(
        self,
        translator: "SilTranslator",
        sentences: Iterable[TSent],
        produce_multiple_translations: bool = False,
    ) -> Iterable[ModelOutputGroup]:
        batch_size: int = self._config.infer["infer_batch_size"]

        current_batch_size = batch_size
        for batch in batch_sentences(sentences, batch_size):
            index = 0
            while index < len(batch):
                effective_size = min(current_batch_size, len(batch) - index)
                sub_batch = batch[index : index + effective_size]
                try:
                    yield from self._translate_sentence_helper(
                        translator,
                        sub_batch,
                        produce_multiple_translations=produce_multiple_translations,
                    )
                    index += effective_size
                except RuntimeError as e:
                    if not _should_reduce_batch_size(e) or current_batch_size <= 1:
                        raise
                    current_batch_size //= 2
                    LOGGER.warning(
                        "OOM during translation inference; reducing batch size to %d and retrying current batch.",
                        current_batch_size,
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def _translate_sentence_helper(
        self,
        translator: "SilTranslator",
        sentences: Iterable[TSent],
        produce_multiple_translations: bool = False,
    ) -> Iterable[ModelOutputGroup]:
        num_drafts = self.get_num_drafts()
        if produce_multiple_translations and num_drafts > 1:
            multiple_translations_method: str = self._config.infer.get("multiple_translations_method")

            if multiple_translations_method == "hybrid":
                beam_search_results: List[List[dict]] = self._translate_with_beam_search(
                    translator,
                    sentences,
                    num_return_sequences=1,
                )

                sampling_results: List[List[dict]] = self._translate_with_sampling(
                    translator,
                    sentences,
                    num_return_sequences=num_drafts - 1,
                )

                # concatenate the beam search results with the sampling results
                yield from [
                    ModelOutputGroup(beam_search_results[i] + sampling_results[i])
                    for i in range(len(beam_search_results))
                ]

            elif multiple_translations_method == "sampling":
                yield from [
                    ModelOutputGroup(result)
                    for result in self._translate_with_sampling(
                        translator,
                        sentences,
                        num_return_sequences=num_drafts,
                    )
                ]

            elif multiple_translations_method == "beam_search":
                yield from [
                    ModelOutputGroup(result)
                    for result in self._translate_with_beam_search(
                        translator,
                        sentences,
                        num_return_sequences=num_drafts,
                    )
                ]

            elif multiple_translations_method == "diverse_beam_search":
                raise RuntimeError(
                    'infer.multiple_translations_method: "diverse_beam_search" is no longer supported, because '
                    'transformers moved group beam search out of the library. Use "hybrid" (the default), '
                    '"beam_search", or "sampling" instead.'
                )
            else:
                LOGGER.error('Unrecognized value for multiple_translations_method: "%s"', multiple_translations_method)

        else:
            yield from [
                ModelOutputGroup([translated_sentence[0]])
                for translated_sentence in self._translate_with_beam_search(
                    translator,
                    sentences,
                    num_return_sequences=1,
                )
            ]

    def _translate_with_beam_search(
        self,
        translator: "SilTranslator",
        sentences: Iterable[TSent],
        num_return_sequences: int = 1,
    ) -> List[List[dict]]:
        num_beams: Optional[int] = self._config.infer.get("num_beams")
        if num_beams is None:
            num_beams = self._config.params.get("generation_num_beams")

        return translator(
            sentences,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )

    def _translate_with_sampling(
        self,
        translator: "SilTranslator",
        sentences: Iterable[TSent],
        num_return_sequences: int = 1,
    ) -> List[List[dict]]:
        temperature: Optional[int] = self._config.infer.get("temperature")

        return translator(
            sentences,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )

    def _create_inference_model(
        self,
        ckpt: Union[CheckpointType, str, int],
        tokenizer: PreTrainedTokenizerBase,
        src_lang: str,
        trg_lang: str,
    ) -> PreTrainedModel:
        if self.has_been_trained():
            checkpoint_path = self.resolve_checkpoint(ckpt).path
            model_name = str(checkpoint_path)
        else:
            LOGGER.warning("Model has no checkpoints. Using base model.")
            model_name = self._config.model

        model: PreTrainedModel = self._pretrained_model_provider.create_model_for_inference(model_name)
        model, tokenizer = self._configure_model(model, tokenizer, src_lang, trg_lang)

        if model.generation_config is not None and (
            model.generation_config.max_length is None or model.generation_config.max_length < 512
        ):
            model.generation_config.max_length = 512

        return model

    def _configure_model(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, src_lang: str, trg_lang: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if trg_lang != "" and model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(trg_lang)

        if self._config.model_name.is_madlad():
            model.config.decoder_start_token_id = tokenizer.pad_token_id
            model.generation_config.decoder_start_token_id = tokenizer.pad_token_id
            model.generation_config.max_length = 256
            model.generation_config.max_new_tokens = 256
            tokenizer.tgt_lang = trg_lang

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
            src_lang != ""
            and trg_lang != ""
            and isinstance(tokenizer, (MBartTokenizer, MBart50Tokenizer, M2M100Tokenizer, NllbTokenizer))
        ):
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = trg_lang

            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token.
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(trg_lang)
            if model.generation_config is not None:
                model.generation_config.forced_bos_token_id = forced_bos_token_id

        if len(tokenizer) > model.get_input_embeddings().weight.size(dim=0):
            # NOTE: This is only a warning because the smoke tests use a mismatched tokenizer and model (intentionally).
            # The long-term fix for this is to use dependency injection for the tokenizer
            LOGGER.warning(
                f"Tokenizer vocab size ({len(tokenizer)}) does not match the model's embedding vocab size "
                f"({model.get_input_embeddings().weight.size(dim=0)}). Ensure you are using the correct "
                f"tokenizer for this checkpoint."
            )

        return model, tokenizer


class SilTranslator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        src_lang: str,
        tgt_lang: str,
    ) -> None:
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __call__(self, sentences: Iterable[Any], **generate_kwargs) -> List[List[dict]]:
        model_inputs = self.preprocess(list(sentences)).to(self.device)
        with torch.no_grad():
            model_outputs = self._forward(model_inputs, **generate_kwargs)
        return self.postprocess(model_outputs)

    def preprocess(self, sentences: List[Any]) -> BatchEncoding:
        # The source language prefix and the forced target language token are configured on the tokenizer and the
        # generation config in _configure_model, so a plain tokenizer call is all that is needed here.
        return self.tokenizer(
            sentences, return_tensors="pt", truncation=TruncationStrategy.DO_NOT_TRUNCATE, padding=True
        )

    def _forward(self, model_inputs, **generate_kwargs):
        in_b, input_length = model_inputs["input_ids"].shape

        config = self.model.generation_config
        generate_kwargs["min_length"] = generate_kwargs.get("min_length", config.min_length)
        generate_kwargs["max_length"] = generate_kwargs.get("max_length", config.max_length)
        output = self.model.generate(
            **model_inputs,
            **generate_kwargs,
            output_scores=True,
            return_dict_in_generate=True,
        )

        output_ids = output.sequences
        output_scores = output.scores
        beam_indices = output.beam_indices if "beam_indices" in output else None
        try:
            transition_scores = self.model.compute_transition_scores(
                output_ids,
                output_scores,
                beam_indices,
                normalize_logits=True,
            )
        except Exception:
            output_ids = output_ids.to("cpu")
            output_scores = tuple(score.to("cpu") for score in output_scores)
            beam_indices = beam_indices.to("cpu") if beam_indices is not None else None
            transition_scores = self.model.compute_transition_scores(
                output_ids,
                output_scores,
                beam_indices,
                normalize_logits=True,
            )
        sequences_scores = getattr(output, "sequences_scores", None)

        out_b, seq_len = output_ids.shape
        n_sequences = out_b // in_b

        ts_len = transition_scores.shape[1]
        if ts_len == seq_len:
            token_logprobs = transition_scores
        elif ts_len == seq_len - 1:
            token_logprobs = torch.cat(
                [
                    torch.zeros(out_b, 1, device=transition_scores.device, dtype=transition_scores.dtype),
                    transition_scores,
                ],
                dim=1,
            )
        else:
            raise RuntimeError(
                f"Unexpected transition_scores length {ts_len} for sequences length {seq_len}. "
                "Cannot align token scores robustly."
            )
        return {
            "output_ids": output_ids.reshape(in_b, n_sequences, seq_len),
            "scores": token_logprobs.reshape(in_b, n_sequences, seq_len),
            "sequences_scores": None if sequences_scores is None else sequences_scores.reshape(in_b, n_sequences),
        }

    def postprocess(self, model_outputs) -> List[List[dict]]:
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer is specified.")

        output_ids: torch.Tensor = model_outputs["output_ids"]
        scores: torch.Tensor = model_outputs["scores"]
        sequences_scores: Optional[torch.Tensor] = model_outputs["sequences_scores"]

        translations: List[List[dict]] = []
        for sentence_index in range(output_ids.size(dim=0)):
            records: List[dict] = []
            for sequence_index in range(output_ids.size(dim=1)):
                sequence_ids = output_ids[sentence_index][sequence_index].tolist()
                token_scores = scores[sentence_index][sequence_index]
                sequence_score = None if sequences_scores is None else sequences_scores[sentence_index][sequence_index]
                translation_text = self.tokenizer.decode(
                    sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                records.append(
                    {
                        "translation_token_ids": sequence_ids,
                        "token_scores": token_scores,
                        "sequence_score": sequence_score,
                        "translation_text": translation_text,
                    }
                )
            translations.append(records)
        return translations


class PretokenizedTranslator(SilTranslator):
    def preprocess(self, sentences: List[Any]) -> BatchEncoding:
        model_inputs = batch_prepare_for_model(self.tokenizer, sentences)
        model_inputs = self.tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
        model_inputs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
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
        tokenizer: PreTrainedTokenizerBase,
        decoder_inputs: DecoderInputs,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        src_noise: List[NoiseMethod] = [],
        return_tensors: str = "pt",
    ):
        # No model is passed on, so that the decoder inputs are built in one place rather than
        # depending on whether the model happens to carry a shift of its own.
        self._data_collator = DataCollatorForSeq2Seq(
            tokenizer, None, padding, max_length, pad_to_multiple_of, label_pad_token_id, return_tensors
        )
        self._decoder_inputs = decoder_inputs
        self._src_noise = src_noise

    def __call__(self, features, return_tensors=None):
        if len(self._src_noise) > 0:
            for feature in features:
                input_ids = feature["input_ids"][:-2]
                for noise_method in self._src_noise:
                    input_ids = noise_method(input_ids)
                feature["input_ids"] = input_ids + feature["input_ids"][-2:]
                feature["attention_mask"] = feature["attention_mask"][: len(feature["input_ids"])]

        batch = self._data_collator(features, return_tensors)
        if batch.get("labels") is not None:
            batch["decoder_input_ids"] = self._decoder_inputs.from_labels(batch["labels"])
        return batch


class SilSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[Seq2SeqTrainingArguments] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[optim.Optimizer], Optional[optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        sequential_sampling: bool = False,
        auto_grad_acc: bool = False,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self._sequential_sampling = sequential_sampling
        self._auto_grac_acc = auto_grad_acc

    def _get_train_sampler(self, train_dataset: Optional[TorchDataset] = None) -> Optional[Sampler]:
        if self._sequential_sampling:
            return None
        return super()._get_train_sampler(train_dataset)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        if self._auto_grac_acc:
            (args if args is not None else self.args).auto_find_batch_size = True
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


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 64, accelerator=None):
    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        last_exception = None

        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.") from last_exception
            open_bars = set(getattr(std_tqdm, "_instances", []))
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if _should_reduce_batch_size(e):
                    last_exception = e
                    _close_orphaned_progress_bars(open_bars)
                    LOGGER.warning(
                        f"Reducing batch size from {batch_size} to {batch_size // 2} after exception: {e}. "
                        f"CUDA memory allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                        f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB, "
                        f"max allocated={torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    accelerator.gradient_accumulation_steps = accelerator.gradient_accumulation_steps * 2
                    kwargs["args"].gradient_accumulation_steps = accelerator.gradient_accumulation_steps
                else:
                    raise

    return decorator


def _close_orphaned_progress_bars(open_bars: Set[Any]) -> None:
    for bar in list(getattr(std_tqdm, "_instances", [])):
        if bar not in open_bars:
            try:
                bar.close()
            except Exception:
                pass


def _should_reduce_batch_size(exception: Exception) -> bool:
    if should_reduce_batch_size(exception):
        return True
    # Check for MIG Out of Memory error. Can remove when should_reduce_batch_size works on MIGs.
    if 'NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp"' in str(exception):
        return True
    return False
