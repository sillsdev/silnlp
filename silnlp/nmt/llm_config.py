"""Fine-tuning and inference for decoder-only LLMs (e.g. TranslateGemma, Hy-MT2).

This module provides a parallel implementation of the :class:`Config`/:class:`NMTModel`
abstractions for decoder-only causal language models, alongside the existing seq2seq
(NLLB/MADLAD) implementation in :mod:`silnlp.nmt.seq2seq_config`. It deliberately
reuses the model-agnostic parts of the pipeline:

* data preparation (``Config.preprocess`` and the corpus writers), by setting
  ``data.tokenize: false`` so the raw detokenized parallel text is used directly and the
  model's own tokenizer handles tokenization;
* evaluation (:mod:`silnlp.nmt.test`) and inference orchestration
  (:mod:`silnlp.nmt.translate`), which depend only on the :class:`NMTModel` interface.

Training supports full fine-tuning as well as low-rank adapters (LoRA and DoRA) via ``peft``,
optionally with 4-bit quantization via ``bitsandbytes`` (QLoRA/QDoRA), selected with
``params.finetune_method``. Adapter hyperparameters live under ``params.adapter``. Prompts are
built with the model's native chat template and a configurable translation instruction.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

from ..common.environment import SilNlpEnv
from ..common.translation_data_structures import DraftGroup, SentenceTranslation, SentenceTranslationGroup
from ..common.translator import generate_confidence_files
from ..common.utils import merge_dict
from .causal_lm_tokenizer import CausalLMTokenizer
from .causal_lm_training_run import CausalLMTrainingRun
from .checkpoints import CheckpointDirectory, CheckpointType
from .config import Config, InferenceModelParams, NMTModel
from .config_keys import DeprecatedAdapterKey, RenamedConfigKeys
from .dictionary_writer import DictionaryWriter, NoDictionaryWriter
from .experiment_files import ExperimentFiles
from .experiment_languages import ExperimentLanguages
from .experiment_settings import EvaluationSettings, TrainerSettings
from .finetune_method import FinetuneMethod
from .finetuning import Finetuning
from .generation_settings import GenerationSettings
from .model_name import ModelName
from .prediction_files import PredictionFile
from .prompt_messages import Language, PromptBuilder
from .seq2seq_config import batch_sentences
from .tokenizer import NullTokenizer, Tokenizer
from .training_arguments import TrainingArgumentsMapping
from .training_data_sets import CausalLMTrainingDataSets
from .vocabulary_builder import NoVocabularyBuilder, VocabularyBuilder

LOGGER = logging.getLogger(__name__)

# Which config sections/keys map onto transformers.TrainingArguments fields. Mirrors
# TRAINING_ARGS_CONFIG_MAPPING in seq2seq_config.py but without the seq2seq-only
# generation keys (generation_max_length, generation_num_beams, predict_with_generate).
_TRAINING_ARGS_CONFIG_MAPPING = {
    "train": {
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "log_level",
        "logging_first_step",
        "logging_steps",
        "logging_strategy",
        "max_steps",
        "num_train_epochs",
        "output_dir",
        "per_device_train_batch_size",
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
        "load_best_model_at_end",
        "metric_for_best_model",
        "per_device_eval_batch_size",
    },
    "params": {
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "learning_rate",
        "lr_scheduler_type",
        "max_grad_norm",
        "optim",
        "warmup_steps",
        "weight_decay",
    },
}


def is_image_text_to_text_model(model_name_or_path: str, trust_remote_code: bool = False) -> bool:
    """Return True if the checkpoint is a multimodal image-text-to-text model."""
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    return type(config) in AutoModelForImageTextToText._model_mapping


class LLMConfig(Config):
    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        DeprecatedAdapterKey().apply_to(config)
        RenamedConfigKeys.of_llm().warn_about(config)
        config = merge_dict(
            {
                "data": {
                    "mirror": False,
                    "seed": 111,
                    # LLMs use their own tokenizer; skip SentencePiece vocab building and
                    # consume the raw (detokenized) parallel text written during preprocessing.
                    "tokenize": False,
                    "aligner": "fast_align",
                    "stats_max_size": 100000,
                    "terms": {"train": False, "categories": "PN", "include_glosses": False, "dictionary": False},
                    "lang_codes": {},
                    "add_new_lang_code": False,
                },
                "train": {
                    "gradient_checkpointing": True,
                    "gradient_checkpointing_kwargs": {"use_reentrant": False},
                    "save_steps": 1000,
                    "per_device_train_batch_size": 4,
                    "save_strategy": "steps",
                    "save_total_limit": 2,
                    "gradient_accumulation_steps": 8,
                    "auto_grad_acc": False,
                    "max_steps": 5000,
                    "train_sampling_strategy": "group_by_length",
                    "output_dir": str(exp_dir / "run"),
                    "log_level": "info",
                },
                "eval": {
                    "eval_strategy": "steps",
                    "eval_steps": 1000,
                    "early_stopping": None,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                    "per_device_eval_batch_size": 4,
                    "multi_ref_eval": False,
                },
                "infer": {
                    "infer_batch_size": 8,
                    "max_new_tokens": 256,
                    "num_beams": 1,
                    "num_drafts": 1,
                    "do_sample": False,
                    "temperature": 0.7,
                },
                "params": {
                    "finetune_method": "qlora",  # full | lora | qlora | dora | qdora
                    "torch_dtype": "bfloat16",
                    "attn_implementation": "sdpa",
                    "trust_remote_code": False,
                    "max_seq_length": 1024,
                    "optim": "adamw_torch",
                    "learning_rate": 0.0002,
                    "lr_scheduler_type": "cosine",
                    "warmup_steps": 150,
                    # Low-rank adapter hyperparameters, shared by all adapter methods
                    # (lora/qlora/dora/qdora). LoRA vs DoRA is selected via finetune_method.
                    "adapter": {
                        "rank": 16,
                        "alpha": 32,
                        "dropout": 0.05,
                        "target_modules": "all-linear",
                        # Layers to train in full (unadapted) alongside the adapters. A list of
                        # module-name suffixes matched against the model's modules; None (or an
                        # empty list) trains only the adapters. Possible choices are:
                        #   "embed_tokens" - the input token-embedding matrix
                        #   "lm_head"      - the output (vocabulary projection) head
                        "modules_to_save": None,
                    },
                    "prompt": {
                        "system_message": "",
                        "instruction_template": (
                            "Translate the following text from {src_lang} to {trg_lang}.\n\n{source}"
                        ),
                    },
                },
                "model": "google/gemma-2-2b-it",
            },
            config,
        )
        super().__init__(exp_dir, config, environment)
        self._hf_tokenizer = CausalLMTokenizer(self.model, self.params["trust_remote_code"])

        if len(self.corpus_inventory.source_isos()) > 1 or len(self.corpus_inventory.target_isos()) > 1:
            raise RuntimeError("LLM experiments only support a single source language and a single target language.")

        self._disable_eval_if_no_val_split()

    def _finetune_method(self) -> FinetuneMethod:
        return FinetuneMethod(self.params["finetune_method"])

    @property
    def finetune_method(self) -> str:
        return str(self._finetune_method())

    @property
    def uses_quantization(self) -> bool:
        return self._finetune_method().uses_quantization()

    @property
    def uses_dora(self) -> bool:
        return self._finetune_method().uses_dora()

    @property
    def adapter(self) -> dict:
        return self.params["adapter"]

    def create_model(
        self,
        mixed_precision: bool = True,
        num_devices: int = 1,
        clearml_queue: Optional[str] = None,
        pretrained_model_provider_factory: Optional["CausalLMProviderFactory"] = None,
    ) -> NMTModel:
        if pretrained_model_provider_factory is None:
            pretrained_model_provider_factory = FileCausalLMProviderFactory()
        trainer_settings = TrainerSettings(self.train)
        provider = pretrained_model_provider_factory.create(
            self.model, self.params, self._finetune_method(), trainer_settings, mixed_precision
        )
        languages = self.create_languages()
        prompts = self.create_prompt_builder()
        return LLMModel(
            CheckpointDirectory(self.model_dir),
            self.infer.get("num_drafts", 1),
            languages,
            self.files,
            self._hf_tokenizer,
            prompts,
            GenerationSettings(self.infer),
            CausalLMTrainingRun(
                provider,
                self._hf_tokenizer,
                CausalLMTrainingDataSets(
                    self.files, self._hf_tokenizer, prompts, languages, self.params["max_seq_length"]
                ),
                Finetuning(self._finetune_method(), self.adapter, trainer_settings),
                EvaluationSettings(self.eval),
                trainer_settings,
                TrainingArgumentsMapping(_TRAINING_ARGS_CONFIG_MAPPING, self.root),
                self.params["torch_dtype"],
                mixed_precision,
                clearml_queue,
            ),
            provider,
            self.data["seed"],
        )

    def create_tokenizer(self) -> Tokenizer:
        # The Config-level Tokenizer is only used by data prep and by test.py to detokenize
        # predictions/references; for LLMs both are raw text, so a no-op tokenizer suffices.
        return NullTokenizer()

    def get_hf_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._hf_tokenizer.load()

    def create_languages(self) -> ExperimentLanguages:
        return ExperimentLanguages(self.data["lang_codes"], self.corpus_inventory)

    def create_prompt_builder(self) -> PromptBuilder:
        return PromptBuilder(ModelName(self.model), self.params["prompt"])

    def create_vocabulary_builder(self) -> VocabularyBuilder:
        return NoVocabularyBuilder()

    def _dictionary_writer(self, tokenizer: Tokenizer) -> DictionaryWriter:
        return NoDictionaryWriter()


@dataclass
class CausalLMProvider:
    """Loads the underlying causal LM for training and inference. Indirected so tests can
    substitute a mock provider (mirrors PreTrainedModelProvider in seq2seq_config.py)."""

    model: str
    params: dict
    finetuning: FinetuneMethod
    trainer_settings: TrainerSettings
    mixed_precision: bool

    def _dtype(self) -> Any:
        if not self.mixed_precision:
            return "auto"
        return getattr(torch, self.params["torch_dtype"], torch.bfloat16)

    def _determine_auto_model_class(self, model_name_or_path: str) -> type:
        if is_image_text_to_text_model(model_name_or_path, self.params["trust_remote_code"]):
            return AutoModelForImageTextToText
        return AutoModelForCausalLM

    def _set_use_cache(self, model: PreTrainedModel, use_cache: bool) -> None:
        # Composite configs (e.g. Gemma3's image-text-to-text wrapper) only expose use_cache on
        # the nested text_config, not on the top-level config, so passing use_cache directly to
        # from_pretrained() leaves it unconsumed there and it gets forwarded as an invalid
        # constructor kwarg to models whose __init__ takes only `config`.
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = use_cache
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = use_cache

    def create_model_for_training(self) -> PreTrainedModel:
        params = self.params
        quantization_config = None
        device_map = None
        if self.finetuning.uses_quantization():
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._dtype(),
                bnb_4bit_use_double_quant=True,
            )
            device_map = {"": 0}
        model_class = self._determine_auto_model_class(self.model)
        model = model_class.from_pretrained(
            self.model,
            quantization_config=quantization_config,
            torch_dtype=self._dtype(),
            attn_implementation=params["attn_implementation"],
            trust_remote_code=params["trust_remote_code"],
            device_map=device_map,
        )
        self._set_use_cache(model, not self.trainer_settings.checkpoints_gradients())
        return model

    def create_model_for_inference(self, checkpoint_path: Optional[Path]) -> PreTrainedModel:
        params = self.params
        load_kwargs = dict(
            torch_dtype=self._dtype(),
            attn_implementation=params["attn_implementation"],
            trust_remote_code=params["trust_remote_code"],
        )
        if checkpoint_path is None:
            model_class = self._determine_auto_model_class(self.model)
            return model_class.from_pretrained(self.model, **load_kwargs)

        if (checkpoint_path / "adapter_config.json").is_file():
            from peft import PeftModel

            model_class = self._determine_auto_model_class(self.model)
            base_model = model_class.from_pretrained(self.model, **load_kwargs)
            base_dtype = next(base_model.parameters()).dtype
            model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
            merged = model.merge_and_unload()
            return merged.to(base_dtype)
        model_class = self._determine_auto_model_class(str(checkpoint_path))
        return model_class.from_pretrained(str(checkpoint_path), **load_kwargs)


class CausalLMProviderFactory:
    def create(
        self,
        model: str,
        params: dict,
        finetuning: FinetuneMethod,
        trainer_settings: TrainerSettings,
        mixed_precision: bool,
    ) -> CausalLMProvider:
        raise NotImplementedError


class FileCausalLMProviderFactory(CausalLMProviderFactory):
    def create(
        self,
        model: str,
        params: dict,
        finetuning: FinetuneMethod,
        trainer_settings: TrainerSettings,
        mixed_precision: bool,
    ) -> CausalLMProvider:
        return CausalLMProvider(model, params, finetuning, trainer_settings, mixed_precision)


class LLMModel(NMTModel):
    def __init__(
        self,
        checkpoints: CheckpointDirectory,
        num_drafts: int,
        languages: ExperimentLanguages,
        files: ExperimentFiles,
        tokenizer: CausalLMTokenizer,
        prompts: PromptBuilder,
        generation: GenerationSettings,
        training_run: CausalLMTrainingRun,
        provider: CausalLMProvider,
        seed: int,
    ) -> None:
        super().__init__(checkpoints, num_drafts)
        self._languages = languages
        self._files = files
        self._tokenizer = tokenizer
        self._prompts = prompts
        self._generation = generation
        self._training_run = training_run
        self._provider = provider
        set_seed(seed)

    def train(self) -> None:
        self._training_run.train()
        self._training_run.save()

    def save_effective_config(self, path: Path) -> None:
        self._training_run.write_effective_config(path)

    def _create_inference_model(self, ckpt: Union[CheckpointType, str, int]) -> PreTrainedModel:
        if self.has_been_trained():
            checkpoint_path = self.resolve_checkpoint(ckpt).path
        else:
            LOGGER.warning("Model has no checkpoints. Using base model.")
            checkpoint_path = None
        model = self._provider.create_model_for_inference(checkpoint_path)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return model

    def _get_inference_model(
        self, ckpt: Union[CheckpointType, str, int], src_lang: str, trg_lang: str
    ) -> PreTrainedModel:
        params = InferenceModelParams(ckpt, src_lang, trg_lang)
        if self._inference_model_params == params and self._cached_inference_model is not None:
            return self._cached_inference_model
        model = self._create_inference_model(ckpt)
        self._cached_inference_model = model
        self._inference_model_params = params
        return model

    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Generator[SentenceTranslationGroup, None, None]:
        src_lang = self._languages.of(src_iso)
        trg_lang = self._languages.of(trg_iso)
        model = self._get_inference_model(ckpt, src_lang.name, trg_lang.name)
        tokenizer = self._tokenizer.load()
        yield from self._generate(model, tokenizer, sentences, src_lang, trg_lang, produce_multiple_translations, False)

    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        tokenizer = self._tokenizer.load()
        src_iso = self._languages.training_source_iso()
        trg_iso = self._languages.training_target_iso()
        src_lang = self._languages.of(src_iso)
        trg_lang = self._languages.of(trg_iso)
        model = self._get_inference_model(ckpt, src_lang.name, trg_lang.name)

        for input_path, translation_path in zip(input_paths, translation_paths):
            file_src_iso, file_trg_iso = self._isos_for_test_file(input_path, src_iso, trg_iso)
            file_src_lang = self._languages.of(file_src_iso)
            file_trg_lang = self._languages.of(file_trg_iso)
            with open(input_path, "r", encoding="utf-8-sig") as src_file:
                sentences = [line.strip() for line in src_file]
            sentence_translation_groups = list(
                self._generate(
                    model,
                    tokenizer,
                    sentences,
                    file_src_lang,
                    file_trg_lang,
                    produce_multiple_translations,
                    save_confidences,
                )
            )
            draft_group = DraftGroup(sentence_translation_groups)
            for draft_index, translated_draft in enumerate(draft_group.get_drafts(), 1):
                if produce_multiple_translations:
                    translation_draft_path = PredictionFile(translation_path).draft(draft_index)
                else:
                    translation_draft_path = translation_path
                with translation_draft_path.open("w", encoding="utf-8", newline="\n") as out_file:
                    out_file.write("\n".join(translated_draft.get_all_tokenized_translations()) + "\n")
                if save_confidences:
                    generate_confidence_files(translated_draft, translation_draft_path)

    def _isos_for_test_file(self, input_path: Path, default_src_iso: str, default_trg_iso: str) -> Tuple[str, str]:
        match = re.match(r"^test\.([a-z]{2,3})\.([a-z]{2,3})\..*", input_path.name)
        if match:
            return match.groups()
        return default_src_iso, default_trg_iso

    def _generate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        sentences: Iterable[str],
        src_lang: Language,
        trg_lang: Language,
        produce_multiple_translations: bool,
        save_confidences: bool,
    ) -> Iterable[SentenceTranslationGroup]:
        tokenizer.padding_side = "left"
        num_drafts = self.get_num_drafts()
        num_return_sequences = num_drafts if (produce_multiple_translations and num_drafts > 1) else 1

        gen_kwargs = self._generation.as_keyword_arguments(num_return_sequences, tokenizer.pad_token_id)

        device = model.device
        for batch in batch_sentences(sentences, self._generation.batch_size()):
            prompts = [
                self._prompts.build(sentence, src_lang, trg_lang).apply_prompt_template(
                    tokenizer, add_generation_prompt=True, tokenize=False
                )
                for sentence in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    output_scores=save_confidences,
                    **gen_kwargs,
                )
            prompt_length = inputs["input_ids"].shape[1]
            generated = output.sequences[:, prompt_length:]

            transition_scores = None
            beam_indices = None
            if save_confidences and getattr(output, "scores", None) is not None:
                beam_indices = getattr(output, "beam_indices", None)
                transition_scores = model.compute_transition_scores(
                    output.sequences, output.scores, beam_indices, normalize_logits=True
                )

            for i in range(len(batch)):
                translations: List[SentenceTranslation] = []
                for j in range(num_return_sequences):
                    seq_index = i * num_return_sequences + j
                    token_ids = generated[seq_index]
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    token_scores: List[float] = []
                    sequence_score: Optional[float] = None
                    if transition_scores is not None:
                        scores_row = transition_scores[seq_index].tolist()
                        if beam_indices is not None:
                            # With beam search, compute_transition_scores() marks padded positions
                            # with a beam index of -1 and a transition score of 0.
                            valid = [s for s, b in zip(scores_row, beam_indices[seq_index].tolist()) if b >= 0]
                        else:
                            valid = [s for s in scores_row if s != float("-inf")]
                        token_scores = valid
                        if len(valid) > 0:
                            sequence_score = sum(valid) / len(valid)
                    translations.append(
                        SentenceTranslation(text, [text], token_scores, sequence_score, starts_with_special_token=False)
                    )
                yield SentenceTranslationGroup(translations)
