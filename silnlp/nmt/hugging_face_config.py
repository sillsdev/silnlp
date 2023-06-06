import json
import re
from contextlib import ExitStack
from copy import deepcopy
from enum import Enum
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union, cast

import datasets.utils.logging as datasets_logging
import evaluate
import numpy as np
import torch
import transformers.utils.logging as transformers_logging
import yaml
from datasets import Dataset
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef
from sacremoses import MosesPunctNormalizer
from torch import Tensor, TensorType
from torch.utils.checkpoint import checkpoint  # noqa: 401
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    NllbTokenizer,
    NllbTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TranslationPipeline,
    set_seed,
)
from transformers.tokenization_utils import BatchEncoding, TruncationStrategy
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import PaddingStrategy, to_py_obj
from transformers.utils.logging import tqdm
from transformers.convert_slow_tokenizer import convert_slow_tokenizer

from ..common.corpus import count_lines, get_terms
from ..common.utils import NoiseMethod, ReplaceRandomToken, Side, create_noise_methods, merge_dict
from .config import CheckpointType, Config, CorpusPair, NMTModel
from .tokenizer import NullTokenizer, Tokenizer


def prepare_decoder_input_ids_from_labels(self: M2M100ForConditionalGeneration, labels: Tensor) -> Tensor:
    # shift ids to the right
    shifted_input_ids = labels.new_zeros(labels.shape)
    shifted_input_ids[:, 1:] = labels[:, :-1].clone()
    # copy lang id from the end to the beginning
    extended_labels = torch.cat((labels, labels.new_full((labels.shape[0], 1), -100)), dim=1)
    lang_index = extended_labels.argmin(dim=1, keepdim=True) - 1
    shifted_input_ids[:, 0] = labels.gather(dim=1, index=lang_index).squeeze()

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


def find_missing_characters(tokenizer: PreTrainedTokenizer, corpus: List[Path]) -> List[str]:
    # create dictionary vocab of tokenizer
    vocab = tokenizer.get_vocab().keys()
    # create set of characters found in corpus
    charset = set()
    for file in corpus:
        with file.open("r", encoding="utf-8-sig") as f:
            charset = charset | set(f.read())
    missing_characters = list(charset - vocab)
    # find characters not in NLLB tokenizer
    return missing_characters


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
            lang_codes: Dict[str, str] = self.data["lang_codes"]
            updated = False
            for iso in self.src_isos | self.trg_isos:
                lang_code = lang_codes.get(iso, iso)
                if lang_code not in self._tokenizer.lang_code_to_id:
                    add_lang_code_to_tokenizer(self._tokenizer, lang_code)
                    updated = True
            missing_characters = find_missing_characters(
                self._tokenizer, list(self.src_file_paths) + list(self.trg_file_paths)
            )
            if missing_characters:
                missing_characters_underscore = ["_" + c for c in missing_characters]
                missing_characters = missing_characters + missing_characters_underscore
                self._tokenizer.add_tokens(missing_characters)
                updated = True
            if updated:
                self._tokenizer.save_pretrained(self.exp_dir)

    def get_tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            if (self.exp_dir / "sentencepiece.bpe.model").is_file() and not (
                self.exp_dir / "tokenizer_config.json"
            ).is_file():
                self._tokenizer = NllbTokenizer.from_pretrained(str(self.exp_dir))
                self._tokenizer = convert_slow_tokenizer(self._tokenizer)
                self._tokenizer = NllbTokenizerFast(tokenizer_object=self._tokenizer)
                self._tokenizer.save_pretrained(str(self.exp_dir))
            else:
                model_name_or_path = (
                    str(self.exp_dir)
                    if (self.exp_dir / "tokenizer_config.json").is_file()
                    else str(Path("silnlp", "assets"))
                    if Path("silnlp", "assets", "tokenizer_config.json")
                    else self.model
                )
                self._tokenizer = NllbTokenizerFast.from_pretrained(model_name_or_path, use_fast=True)
        self._tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self._build_vocabs()
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
        )
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(self._config.model, config=model_config)
        tokenizer = self._config.get_tokenizer()

        model.resize_token_embeddings(len(tokenizer))

        tokenizer.src_lang = self._config.val_src_lang
        tokenizer.tgt_lang = self._config.val_trg_lang
        trg_lang_id = tokenizer.convert_tokens_to_ids(self._config.val_trg_lang)
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

    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        vref_paths: Optional[List[Path]] = None,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        checkpoint_path, _ = self.get_checkpoint_path(ckpt)
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_path))
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
        model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
        for prediction in tqdm(
            self._translate_sentences(tokenizer, pipeline, sentences, vrefs),
            total=len(sentences),
            unit="ex",
        ):
            yield prediction["translation_text"]

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
            step = int(ckpt.name[11:])
        elif ckpt is CheckpointType.LAST:
            ckpt_path = Path(get_last_checkpoint(self._config.model_dir))
            step = int(ckpt.name[11:])
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
        merge_dict(args, {"fp16": self._mixed_precision})
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
