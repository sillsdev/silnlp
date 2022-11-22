import json
import re
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import datasets.utils.logging as datasets_logging
import evaluate
import numpy as np
import transformers.utils.logging as transformers_logging
import yaml
from datasets import Dataset
from sacremoses import MosesPunctNormalizer
from torch import Tensor, TensorType
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TranslationPipeline,
    set_seed,
)
from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right
from transformers.tokenization_utils import BatchEncoding, TruncationStrategy
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import to_py_obj
from transformers.utils.logging import tqdm

from ..common.corpus import count_lines
from ..common.environment import SIL_NLP_ENV
from ..common.utils import Side, merge_dict
from .config import CheckpointType, Config, NMTModel
from .tokenizer import NullTokenizer, Tokenizer


def prepare_decoder_input_ids_from_labels(self: M2M100ForConditionalGeneration, labels: Tensor) -> Tensor:
    return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)


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


def get_checkpoint_path(model_dir: Path, checkpoint: Union[CheckpointType, str, int]) -> Tuple[Path, int]:
    step: Optional[int] = None
    if isinstance(checkpoint, str):
        checkpoint = checkpoint.lower()
        if "avg" in checkpoint:
            checkpoint = CheckpointType.AVERAGE
        elif "best" in checkpoint:
            checkpoint = CheckpointType.BEST
        elif "last" in checkpoint:
            checkpoint = CheckpointType.LAST
        else:
            step = int(checkpoint)
            checkpoint = CheckpointType.OTHER
    if checkpoint is CheckpointType.BEST:
        ckpt = get_best_checkpoint(model_dir)
        step = int(ckpt.name[11:])
    elif checkpoint is CheckpointType.LAST:
        ckpt = Path(get_last_checkpoint(model_dir))
        step = int(ckpt.name[11:])
    elif checkpoint is CheckpointType.OTHER and step is not None:
        ckpt = model_dir / f"checkpoint-{step}"
    else:
        raise ValueError(f"Unsupported checkpoint type: {checkpoint}.")
    if ckpt is not None:
        SIL_NLP_ENV.copy_experiment_from_bucket(SIL_NLP_ENV.get_source_experiment_path(ckpt))
    return ckpt, step


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
    if isinstance(tokenizer, M2M100Tokenizer):
        tokenizer.lang_token_to_id[lang_code] = lang_id
        tokenizer.id_to_lang_token[lang_id] = lang_code


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
                    "save_total_limit": 1,
                    "gradient_accumulation_steps": 4,
                    "max_steps": 100000,
                    "group_by_length": True,
                    "output_dir": str(exp_dir / "run"),
                    "delete_checkpoint_optimizer_state": True,
                    "delete_checkpoint_tokenizer": True,
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
                },
                "infer": {"infer_batch_size": 32},
                "params": {"optim": "adamw_torch"},
            },
            config,
        )
        self._tokenizer: Optional[PreTrainedTokenizer] = None

        super().__init__(exp_dir, config)

        if len(self.src_isos) > 1 or len(self.trg_isos) > 1 or self.mirror:
            raise RuntimeError("HuggingFace does not support training multilingual models.")

    @property
    def model_dir(self) -> Path:
        return Path(self.train["output_dir"])

    @property
    def src_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.default_src_iso, self.default_src_iso)

    @property
    def trg_lang(self) -> str:
        lang_codes: Dict[str, str] = self.data["lang_codes"]
        return lang_codes.get(self.default_trg_iso, self.default_trg_iso)

    def create_model(self, mixed_precision: bool = False, num_devices: int = 1) -> NMTModel:
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

    def _build_vocabs(self) -> None:
        if self.data["add_new_lang_code"]:
            tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
            lang_codes: Dict[str, str] = self.data["lang_codes"]
            updated = False
            for iso in self.src_isos | self.trg_isos:
                lang_code = lang_codes.get(iso, iso)
                if lang_code not in tokenizer.lang_code_to_id:
                    add_lang_code_to_tokenizer(tokenizer, lang_code)
                    updated = True
            if updated:
                tokenizer.save_pretrained(self.exp_dir)

    def get_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            model_name_or_path = str(self.exp_dir) if (self.exp_dir / "tokenizer_config.json").is_file() else self.model
            self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            self._tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return self._tokenizer


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


class HuggingFaceNMTModel(NMTModel):
    def __init__(self, config: HuggingFaceConfig, mixed_precision: bool, num_devices: int) -> None:
        self._config = config
        self._mixed_precision = mixed_precision
        set_seed(self._config.data["seed"])

    def train(self) -> None:
        training_args = self._create_training_arguments()

        log_level = training_args.get_process_log_level()
        datasets_logging.set_verbosity(log_level)
        transformers_logging.set_verbosity(log_level)
        transformers_logging.enable_default_handler()
        transformers_logging.enable_explicit_format()

        model_config = AutoConfig.from_pretrained(
            self._config.model, use_cache=not training_args.gradient_checkpointing
        )
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(self._config.model, config=model_config)
        tokenizer = self._config.get_tokenizer()

        model.resize_token_embeddings(len(tokenizer))

        tokenizer.src_lang = self._config.src_lang
        tokenizer.tgt_lang = self._config.trg_lang
        trg_lang_id = tokenizer.convert_tokens_to_ids(self._config.trg_lang)
        model.config.decoder_start_token_id = trg_lang_id
        model.config.forced_bos_token_id = trg_lang_id

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

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        metric = evaluate.load("sacrebleu")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [[label.strip()] for label in decoded_labels]

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, lowercase=True)
            result = {"bleu": result["score"]}

            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

        trainer = Seq2SeqTrainer(
            model,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics=compute_metrics,
        )
        early_stopping: Optional[dict] = self._config.eval["early_stopping"]
        if early_stopping is not None:
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

    def translate_text_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        ref_paths: Optional[List[Path]] = None,
        checkpoint: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        checkpoint_path, _ = get_checkpoint_path(self._config.model_dir, checkpoint)
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_path))
        tokenizer = self._config.get_tokenizer()
        pipeline = PretokenizedTranslationPipeline(
            model=model, tokenizer=tokenizer, src_lang=self._config.src_lang, tgt_lang=self._config.trg_lang, device=0
        )
        batch_size: int = self._config.infer["infer_batch_size"]
        num_beams: Optional[int] = self._config.infer.get("num_beams")
        if num_beams is None:
            num_beams = self._config.params.get("generation_num_beams")
        for input_path, translation_path in zip(input_paths, translation_paths):
            if not isinstance(input_path, Path):
                input_path = input_path[0]
            length = count_lines(input_path)
            with input_path.open("r", encoding="utf-8-sig") as src_file, translation_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as out_file:
                sentences = (line.strip().split() for line in src_file)
                for prediction in tqdm(
                    pipeline(
                        sentences, return_text=False, return_tensors=True, batch_size=batch_size, num_beams=num_beams
                    ),
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
        refs: Optional[Iterable[str]] = None,
        checkpoint: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Iterable[str]:
        checkpoint_path, _ = get_checkpoint_path(self._config.model_dir, checkpoint)
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_path))
        tokenizer = self._config.get_tokenizer()
        lang_codes: Dict[str, str] = self._config.data["lang_codes"]
        pipeline = TranslationPipeline(
            model=model,
            tokenizer=tokenizer,
            src_lang=lang_codes.get(src_iso, src_iso),
            tgt_lang=lang_codes.get(trg_iso, trg_iso),
            device=0,
        )
        batch_size: int = self._config.infer["infer_batch_size"]
        num_beams: Optional[int] = self._config.infer.get("num_beams")
        if num_beams is None:
            num_beams = self._config.params.get("generation_num_beams")
        sentences = [line if isinstance(line, str) else line[0] for line in sentences]
        for prediction in tqdm(
            pipeline(sentences, batch_size=batch_size, num_beams=num_beams), total=len(sentences), unit="ex"
        ):
            yield prediction["translation_text"]

    def get_checkpoint_step(self, checkpoint: Union[CheckpointType, str, int]) -> int:
        return get_checkpoint_path(self._config.model_dir, checkpoint)[1]

    def _create_training_arguments(self) -> Seq2SeqTrainingArguments:
        parser = HfArgumentParser(Seq2SeqTrainingArguments)
        args: dict = {}
        for section, params in TRAINING_ARGS_CONFIG_MAPPING.items():
            section_config: dict = self._config.root[section]
            for param in params:
                if param in section_config:
                    args[param] = section_config[param]
        merge_dict(args, {"fp16": self._mixed_precision})
        return parser.parse_dict(args)[0]


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
        return " ".join(tokens)

    def normalize(self, side: Side, line: str) -> str:
        return self.detokenize(self.tokenize(side, line))

    def detokenize(self, line: str) -> str:
        tokens = line.split()
        tokens = [p for p in tokens if p not in self._all_special_tokens]
        return self._tokenizer.clean_up_tokenization(self._tokenizer.convert_tokens_to_string(tokens))


class PretokenizedTranslationPipeline(TranslationPipeline):
    def preprocess(self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None):
        model_inputs = batch_prepare_for_model(self.tokenizer, args, return_tensors=self.framework)
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        model_inputs["forced_bos_token_id"] = tgt_lang_id
        return model_inputs
