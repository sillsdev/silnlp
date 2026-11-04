import gc
import logging
import os
import re
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, TypeVar, Union, cast

import safetensors.torch
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from transformers.modeling_utils import unwrap_model
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import SAFE_WEIGHTS_NAME
from transformers.utils.generic import to_py_obj
from transformers.utils.logging import tqdm

from ..common.corpus import count_lines
from ..common.environment import SilNlpEnv
from ..common.translation_data_structures import DraftGroup, SentenceTranslation, SentenceTranslationGroup
from ..common.translator import generate_confidence_files
from ..common.utils import merge_dict
from .batch_size import indicates_out_of_memory
from .checkpoints import CheckpointDirectory, CheckpointType
from .config import (
    Config,
    InferenceModelParams,
    NMTModel,
)
from .seq2seq_training_run import Seq2SeqTrainingRun
from .training_data_sets import TokenizedBatchEncoder
from .training_arguments import TrainingArgumentsMapping
from .translation_settings import CheckpointRetention, ModelSettings, TranslationSettings
from .config_keys import RenamedConfigKeys
from .dictionary_writer import DictionaryWriter, TermDictionaryWriter
from .experiment_languages import ExperimentLanguages
from .huggingface_tokenizer import HuggingFaceTokenizer, PunctuationNormalizingTokenizer
from .experiment_settings import EvaluationSettings, TrainerSettings, TrainingSettings
from .model_name import ModelName
from .parent_model import ParentModel
from .pretrained_model_loader import PretrainedModelLoader
from .pretrained_tokenizer import PretrainedTokenizer
from .tokenizer_settings import TokenizerSettings, TokenizerSource
from .vocabulary import LanguageCodes, MissingTokens, TokenizerVocabularyBuilder
from .vocabulary_builder import NoVocabularyBuilder, VocabularyBuilder
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
        self,
        model_settings: "ModelSettings",
        model_name: ModelName,
        pretrained_tokenizer: PretrainedTokenizer,
        languages: ExperimentLanguages,
        mixed_precision: bool = False,
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
        self,
        model_settings: "ModelSettings",
        model_name: ModelName,
        pretrained_tokenizer: PretrainedTokenizer,
        languages: ExperimentLanguages,
        mixed_precision: bool = False,
    ) -> PreTrainedModelProvider:
        dtype = torch.bfloat16 if model_name.is_t5() else torch.float16
        if not mixed_precision:
            dtype = "auto"
        return FilePreTrainedModelProvider(model_settings.attention_implementation(), dtype)


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
        self._pretrained_tokenizer = PretrainedTokenizer(self._tokenizer_source, self.model_name, self.exp_dir)
        self._hugging_face_tokenizer = HuggingFaceTokenizer(
            self._pretrained_tokenizer,
            self.data["lang_codes"],
            self.train["max_source_length"],
            self.train["max_target_length"],
        )

    def create_languages(self) -> ExperimentLanguages:
        return ExperimentLanguages(self.data["lang_codes"], self.inventory)

    def create_model(
        self,
        mixed_precision: bool = True,
        num_devices: int = 1,
        clearml_queue: Optional[str] = None,
        pretrained_model_provider_factory: PreTrainedModelProviderFactory = FilePreTrainedModelProviderFactory(),
    ) -> NMTModel:
        provider = pretrained_model_provider_factory.create_pretrained_model_provider(
            ModelSettings(self.params),
            self.model_name,
            self._pretrained_tokenizer,
            self.create_languages(),
            mixed_precision,
        )
        model_loader = PretrainedModelLoader(
            provider,
            self.model,
            self.model_name,
            self._pretrained_tokenizer,
            self._tokenizer_settings,
            ModelSettings(self.params),
            CheckpointDirectory(self.model_dir),
            num_devices,
        )
        return Seq2SeqNMTModel(
            CheckpointDirectory(self.model_dir),
            self.infer.get("num_drafts", 1),
            self.create_languages(),
            self._pretrained_tokenizer,
            model_loader,
            TranslationSettings(self.infer, self.params),
            Seq2SeqTrainingRun(
                model_loader,
                self._pretrained_tokenizer,
                self.files,
                self.create_languages(),
                EvaluationSettings(self.eval),
                TrainerSettings(self.train),
                CheckpointRetention(self.train),
                TrainingArgumentsMapping(_TRAINING_ARGS_CONFIG_MAPPING, self.root),
                self.model_name.is_t5(),
                mixed_precision,
                clearml_queue,
            ),
            self.data["seed"],
        )

    def create_tokenizer(self) -> Tokenizer:
        if not self.data["tokenize"]:
            return NullTokenizer()
        return self._hugging_face_tokenizer

    def create_vocabulary_builder(self) -> VocabularyBuilder:
        if not self.data["tokenize"]:
            return NoVocabularyBuilder()
        return TokenizerVocabularyBuilder(
            self._pretrained_tokenizer,
            self.create_tokenizer(),
            MissingTokens(
                self._pretrained_tokenizer,
                self._hugging_face_tokenizer,
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
        )

    def get_or_create_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._pretrained_tokenizer.build()

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._pretrained_tokenizer.load()

    def _dictionary_writer(self, tokenizer: Tokenizer) -> DictionaryWriter:
        return TermDictionaryWriter(
            self.files, tokenizer, self._term_categories(), self._gloss_language(), self._environment
        )


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
        checkpoints: CheckpointDirectory,
        num_drafts: int,
        languages: ExperimentLanguages,
        pretrained_tokenizer: PretrainedTokenizer,
        model_loader: PretrainedModelLoader,
        translation: TranslationSettings,
        training_run: Seq2SeqTrainingRun,
        seed: int,
    ) -> None:
        super().__init__(checkpoints, num_drafts)
        self._languages = languages
        self._pretrained_tokenizer = pretrained_tokenizer
        self._model_loader = model_loader
        self._translation = translation
        self._training_run = training_run
        set_seed(seed)

    def train(self) -> None:
        self._training_run.train()
        self._training_run.save()

    def save_effective_config(self, path: Path) -> None:
        self._training_run.write_effective_config(path)

    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        tokenizer = self._pretrained_tokenizer.load()
        model = self._model_loader.for_inference(ckpt, self._languages.test_source(), self._languages.test_target())
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
            src_lang = self._languages.name_of(src_iso)
            trg_lang = self._languages.name_of(trg_iso)
        else:
            src_lang = self._languages.test_source()
            trg_lang = self._languages.test_target()

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
        src_lang = self._languages.name_of(src_iso)
        trg_lang = self._languages.name_of(trg_iso)
        inference_model_params = InferenceModelParams(ckpt, src_lang, trg_lang)
        tokenizer = self._pretrained_tokenizer.load()
        if self._inference_model_params == inference_model_params and self._cached_inference_model is not None:
            model = self._cached_inference_model
        else:
            model = self._cached_inference_model = self._model_loader.for_inference(ckpt, src_lang, trg_lang)
            self._inference_model_params = inference_model_params

        # The tokenizer isn't wrapped until after the inference model is built,
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

    def _translate_sentences(
        self,
        translator: "SilTranslator",
        sentences: Iterable[TSent],
        produce_multiple_translations: bool = False,
    ) -> Iterable[ModelOutputGroup]:
        batch_size = self._translation.batch_size()

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
                    if not indicates_out_of_memory(e) or current_batch_size <= 1:
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
            multiple_translations_method: str = self._translation.multiple_translations_method()

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
        return translator(
            sentences,
            num_beams=self._translation.beam_count(),
            num_return_sequences=num_return_sequences,
        )

    def _translate_with_sampling(
        self,
        translator: "SilTranslator",
        sentences: Iterable[TSent],
        num_return_sequences: int = 1,
    ) -> List[List[dict]]:
        return translator(
            sentences,
            do_sample=True,
            temperature=self._translation.temperature(),
            num_return_sequences=num_return_sequences,
        )


class SilTranslator:
    """Duplicates sil-machine's HuggingFaceNmtEngine, which cannot replace it until that engine takes
    the tokenizer it is given rather than loading one from the model path: checkpoints hold no
    tokenizer when delete_checkpoint_tokenizer is set, which is the default. Its output_attentions
    default of True, which forces the slower eager attention, wants changing there too."""

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
        model_inputs = TokenizedBatchEncoder(self.tokenizer).encode(sentences)
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
