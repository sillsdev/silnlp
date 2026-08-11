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
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from jinja2.exceptions import UndefinedError
from machine.corpora import TextFileTextCorpus
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..common.environment import SilNlpEnv
from ..common.translation_data_structures import DraftGroup, SentenceTranslation, SentenceTranslationGroup
from ..common.translator import generate_confidence_files
from ..common.utils import merge_dict
from .config import (
    CheckpointType,
    Config,
    InferenceModelParams,
    NMTModel,
    collect_training_args,
    find_last_checkpoint,
    warn_about_renamed_keys,
    write_effective_config,
)
from .corpora import DataFile
from .seq2seq_config import batch_sentences, find_executable_batch_size
from .tokenizer import NullTokenizer, Tokenizer

LOGGER = logging.getLogger(__name__)

# Which config sections/keys map onto transformers.TrainingArguments fields. Mirrors
# TRAINING_ARGS_CONFIG_MAPPING in seq2seq_config.py but without the seq2seq-only
# generation keys (generation_max_length, generation_num_beams, predict_with_generate).
TRAINING_ARGS_CONFIG_MAPPING = {
    "train": {
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "log_level",
        "logging_dir",
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
        "warmup_ratio",
        "warmup_steps",
        "weight_decay",
    },
}

# Config keys renamed in transformers 5.0, can remove after users have gotten used to the transition
RENAMED_CONFIG_KEYS = {
    "train": {"group_by_length": "train_sampling_strategy"},
}

LABEL_PAD_TOKEN_ID = -100

FULL_FINETUNE_METHOD = "full"
ADAPTER_METHODS = ("lora", "qlora", "dora", "qdora")
QUANTIZED_METHODS = ("qlora", "qdora")
DORA_METHODS = ("dora", "qdora")
VALID_FINETUNE_METHODS = (FULL_FINETUNE_METHOD,) + ADAPTER_METHODS

# TranslateGemma's chat template rejects a plain-text user turn: it requires `content` to be a
# single-item list of {type, source_lang_code, target_lang_code, text|image}, and it renders the
# natural-language instruction itself from a fixed table of ~55 supported language codes.
TRANSLATE_GEMMA_MODEL_PREFIXES = ("google/translate-gemma", "google/translategemma")


def is_image_text_to_text_model(model_name_or_path: str, trust_remote_code: bool = False) -> bool:
    """Return True if the checkpoint is a multimodal image-text-to-text model."""
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    return type(config) in AutoModelForImageTextToText._model_mapping


def build_generation_kwargs(infer: dict, num_return_sequences: int, pad_token_id: Optional[int]) -> Dict[str, Any]:
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": infer["max_new_tokens"],
        "num_return_sequences": num_return_sequences,
        "pad_token_id": pad_token_id,
    }
    if infer.get("do_sample"):
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = infer["temperature"]
    else:
        num_beams: int = infer["num_beams"]
        if num_return_sequences > num_beams:
            raise RuntimeError(
                f"Beam search cannot return {num_return_sequences} drafts with num_beams set to {num_beams}. "
                "Increase num_beams to at least num_drafts or set do_sample to true."
            )
        gen_kwargs["num_beams"] = num_beams
    return gen_kwargs


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool,
    tokenize: bool,
) -> Union[str, List[int]]:
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize, return_dict=False
    )


@dataclass(frozen=True)
class Language:
    iso: str
    name: str


@dataclass
class PromptMessages:
    """The chat messages for a translation prompt: an optional system message, a plain-text
    user instruction, and, for training examples, the target translation as the assistant turn."""

    system_message: str
    instruction: str
    target: Optional[str] = None

    def to_chat_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": self.instruction})
        if self.target is not None:
            messages.append({"role": "assistant", "content": self.target})
        return messages

    def to_folded_chat_messages(self) -> List[Dict[str, str]]:
        """Chat messages with the system message folded into the user turn, for chat
        templates that reject a separate system role (e.g. Gemma)."""
        instruction = f"{self.system_message}\n\n{self.instruction}" if self.system_message else self.instruction
        messages: List[Dict[str, str]] = [{"role": "user", "content": instruction}]
        if self.target is not None:
            messages.append({"role": "assistant", "content": self.target})
        return messages

    def to_plain_text(self) -> str:
        return "".join(f"{m['content']}\n" for m in self.to_chat_messages())

    def apply_prompt_template(
        self, tokenizer: PreTrainedTokenizerBase, add_generation_prompt: bool, tokenize: bool
    ) -> Union[str, List[int]]:
        """Apply the model's chat template, with fallbacks for templates that lack a
        system role and for base checkpoints with no chat template at all."""
        if tokenizer.chat_template is not None:
            try:
                return apply_chat_template(tokenizer, self.to_chat_messages(), add_generation_prompt, tokenize)
            except Exception:
                # Some chat templates (e.g. Gemma) reject a separate system role; fold the
                # system message into the first user turn and retry.
                if self.system_message:
                    return apply_chat_template(
                        tokenizer, self.to_folded_chat_messages(), add_generation_prompt, tokenize
                    )
                raise

        LOGGER.warning(
            "Tokenizer for %s has no chat template; falling back to a plain text prompt.", tokenizer.name_or_path
        )
        text = self.to_plain_text()
        if tokenize:
            return tokenizer(text, add_special_tokens=True)["input_ids"]
        return text


@dataclass(init=False)
class TranslateGemmaPromptMessages(PromptMessages):
    """TranslateGemma's chat template rejects a plain-text user turn: it requires ``content``
    to be a single-item list of {type, source_lang_code, target_lang_code, text}, so this
    subclass carries that structured content instead of a plain-text instruction. It has no
    system message, and -- since TranslateGemma always ships a chat template -- never needs
    the system-message-folding or no-chat-template fallbacks of the base class."""

    source_language: Language
    target_language: Language
    text: str

    def __init__(
        self, source_language: Language, target_language: Language, text: str, target: Optional[str] = None
    ) -> None:
        super().__init__(system_message="", instruction="", target=target)
        self.source_language = source_language
        self.target_language = target_language
        self.text = text

    def to_chat_messages(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": self.source_language.iso,
                        "target_lang_code": self.target_language.iso,
                        "text": self.text,
                    }
                ],
            }
        ]
        if self.target is not None:
            messages.append({"role": "assistant", "content": self.target})
        return messages

    def to_folded_chat_messages(self) -> List[Dict[str, str]]:
        raise NotImplementedError("TranslateGemma's structured content is never folded.")

    def to_plain_text(self) -> str:
        raise NotImplementedError("TranslateGemma always has a chat template; there is no plain-text fallback.")

    def apply_prompt_template(
        self, tokenizer: PreTrainedTokenizerBase, add_generation_prompt: bool, tokenize: bool
    ) -> Union[str, List[int]]:
        if tokenizer.chat_template is not None:
            try:
                return apply_chat_template(tokenizer, self.to_chat_messages(), add_generation_prompt, tokenize)
            except UndefinedError:
                # TranslateGemma's template only recognizes its fixed ~55-language lookup table
                # and raises UndefinedError for any other code -- which is the common case when
                # fine-tuning to extend coverage to a new language. Render the same instruction
                # ourselves, using our own configured language name instead of the template's.
                text = self._render_fallback_prompt(tokenizer, add_generation_prompt)
                if tokenize:
                    return tokenizer(text, add_special_tokens=False)["input_ids"]
                return text

        LOGGER.warning(
            "Tokenizer for %s has no chat template; falling back to a plain text prompt.", tokenizer.name_or_path
        )
        text = self._render_fallback_prompt(tokenizer, add_generation_prompt)
        if tokenize:
            return tokenizer(text, add_special_tokens=False)["input_ids"]
        return text

    def _render_fallback_prompt(self, tokenizer: PreTrainedTokenizerBase, add_generation_prompt: bool) -> str:
        """Reimplementation of TranslateGemma's chat template, minus its language-code lookup
        table, for language codes that table doesn't recognize (see apply_prompt_template)."""
        src, trg = self.source_language, self.target_language
        instruction = (
            f"You are a professional {src.name} ({src.iso}) to {trg.name} ({trg.iso}) translator. Your goal is "
            f"to accurately convey the meaning and nuances of the original {src.name} text while adhering to "
            f"{trg.name} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {trg.name} translation, without any additional explanations or commentary. "
            f"Please translate the following {src.name} text into {trg.name}:\n\n\n{self.text.strip()}"
        )
        text = (tokenizer.bos_token or "") + f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        if self.target is not None:
            text += f"<start_of_turn>model\n{self.target.strip()}<end_of_turn>\n"
        if add_generation_prompt:
            text += "<start_of_turn>model\n"
        return text


class LLMConfig(Config):
    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        self._normalize_deprecated_keys(config)
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
                    "warmup_ratio": 0.03,
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
        self._hf_tokenizer: Optional[PreTrainedTokenizerBase] = None

        super().__init__(exp_dir, config, environment)

        if len(self.src_isos) > 1 or len(self.trg_isos) > 1:
            raise RuntimeError("LLM experiments only support a single source language and a single target language.")

        self._disable_eval_if_no_val_split()

    @staticmethod
    def _normalize_deprecated_keys(config: dict) -> None:
        # ``params.lora`` was renamed to ``params.adapter`` when DoRA was added, since the same
        # hyperparameters now back both LoRA and DoRA. Accept the old key for backward compatibility.
        params = config.get("params")
        if isinstance(params, dict) and "lora" in params and "adapter" not in params:
            LOGGER.warning("params.lora is deprecated; rename it to params.adapter.")
            params["adapter"] = params.pop("lora")
        warn_about_renamed_keys(config, RENAMED_CONFIG_KEYS)

    @property
    def finetune_method(self) -> str:
        method = self.params["finetune_method"].lower()
        if method not in VALID_FINETUNE_METHODS:
            raise ValueError(f"Unknown finetune_method '{method}'. Valid options: {', '.join(VALID_FINETUNE_METHODS)}.")
        return method

    @property
    def uses_quantization(self) -> bool:
        return self.finetune_method in QUANTIZED_METHODS

    @property
    def uses_dora(self) -> bool:
        return self.finetune_method in DORA_METHODS

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
        return LLMModel(self, mixed_precision, num_devices, clearml_queue, pretrained_model_provider_factory)

    def create_tokenizer(self) -> Tokenizer:
        # The Config-level Tokenizer is only used by data prep and by test.py to detokenize
        # predictions/references; for LLMs both are raw text, so a no-op tokenizer suffices.
        return NullTokenizer()

    def get_hf_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._hf_tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=self.params["trust_remote_code"])
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._hf_tokenizer = tokenizer
        return self._hf_tokenizer

    def lang_name(self, iso: str) -> str:
        return self.data["lang_codes"].get(iso, iso)

    def language(self, iso: str) -> Language:
        return Language(iso=iso, name=self.lang_name(iso))

    @property
    def train_src_iso(self) -> str:
        return self.default_test_src_iso or (next(iter(self.src_isos)) if len(self.src_isos) > 0 else "")

    @property
    def train_trg_iso(self) -> str:
        return self.default_test_trg_iso or (next(iter(self.trg_isos)) if len(self.trg_isos) > 0 else "")

    def build_prompt_messages(
        self, source: str, src_lang: Language, trg_lang: Language, target: Optional[str] = None
    ) -> PromptMessages:
        if self.model.lower().startswith(TRANSLATE_GEMMA_MODEL_PREFIXES):
            return TranslateGemmaPromptMessages(
                source_language=src_lang, target_language=trg_lang, text=source, target=target
            )

        prompt_config: dict = self.params["prompt"]
        instruction = prompt_config["instruction_template"].format(
            src_lang=src_lang.name, trg_lang=trg_lang.name, source=source
        )
        return PromptMessages(prompt_config.get("system_message", ""), instruction, target)

    def _build_vocabs(self, stats: bool = False) -> None:
        # No vocabulary surgery for decoder-only LLMs; they use their own tokenizer.
        return

    def _write_dictionary(
        self,
        tokenizer: Tokenizer,
        src_terms_files: List[Tuple[DataFile, List[str]]],
        trg_terms_files: List[Tuple[DataFile, List[str]]],
    ) -> int:
        return 0


@dataclass
class CausalLMProvider:
    """Loads the underlying causal LM for training and inference. Indirected so tests can
    substitute a mock provider (mirrors PreTrainedModelProvider in seq2seq_config.py)."""

    config: "LLMConfig"
    mixed_precision: bool

    def _dtype(self) -> Any:
        if not self.mixed_precision:
            return "auto"
        return getattr(torch, self.config.params["torch_dtype"], torch.bfloat16)

    def _determine_auto_model_class(self, model_name_or_path: str) -> type:
        if is_image_text_to_text_model(model_name_or_path, self.config.params["trust_remote_code"]):
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
        params = self.config.params
        quantization_config = None
        device_map = None
        if self.config.uses_quantization:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._dtype(),
                bnb_4bit_use_double_quant=True,
            )
            device_map = {"": 0}
        model_class = self._determine_auto_model_class(self.config.model)
        model = model_class.from_pretrained(
            self.config.model,
            quantization_config=quantization_config,
            torch_dtype=self._dtype(),
            attn_implementation=params["attn_implementation"],
            trust_remote_code=params["trust_remote_code"],
            device_map=device_map,
        )
        self._set_use_cache(model, not self.config.train["gradient_checkpointing"])
        return model

    def create_model_for_inference(self, checkpoint_path: Optional[Path]) -> PreTrainedModel:
        params = self.config.params
        load_kwargs = dict(
            torch_dtype=self._dtype(),
            attn_implementation=params["attn_implementation"],
            trust_remote_code=params["trust_remote_code"],
        )
        if checkpoint_path is None:
            model_class = self._determine_auto_model_class(self.config.model)
            return model_class.from_pretrained(self.config.model, **load_kwargs)

        if (checkpoint_path / "adapter_config.json").is_file():
            from peft import PeftModel

            model_class = self._determine_auto_model_class(self.config.model)
            base_model = model_class.from_pretrained(self.config.model, **load_kwargs)
            base_dtype = next(base_model.parameters()).dtype
            model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
            merged = model.merge_and_unload()
            return merged.to(base_dtype)
        model_class = self._determine_auto_model_class(str(checkpoint_path))
        return model_class.from_pretrained(str(checkpoint_path), **load_kwargs)


class CausalLMProviderFactory:
    def create(self, config: "LLMConfig", mixed_precision: bool) -> CausalLMProvider:
        raise NotImplementedError


class FileCausalLMProviderFactory(CausalLMProviderFactory):
    def create(self, config: "LLMConfig", mixed_precision: bool) -> CausalLMProvider:
        return CausalLMProvider(config, mixed_precision)


@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = LABEL_PAD_TOKEN_ID
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        pad_token_id = self.tokenizer.pad_token_id
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []
        for feature in features:
            ids = feature["input_ids"]
            mask = feature.get("attention_mask", [1] * len(ids))
            label = feature["labels"]
            pad_len = max_length - len(ids)
            # Right padding for training.
            input_ids.append(ids + [pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(label + [self.label_pad_token_id] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SilCausalTrainer(Trainer):
    def __init__(self, *args, auto_grad_acc: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_grad_acc = auto_grad_acc

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        if self._auto_grad_acc:
            (args if args is not None else self.args).auto_find_batch_size = True
            inner_training_loop = find_executable_batch_size(super()._inner_training_loop, batch_size, self.accelerator)
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        return super()._inner_training_loop(
            batch_size=batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )


class LLMModel(NMTModel):
    def __init__(
        self,
        config: LLMConfig,
        mixed_precision: bool,
        num_devices: int,
        clearml_queue: Optional[str] = None,
        pretrained_model_provider_factory: CausalLMProviderFactory = FileCausalLMProviderFactory(),
    ) -> None:
        super().__init__(config)
        self._config: LLMConfig = config
        self._mixed_precision = mixed_precision
        self._num_devices = num_devices
        self._clearml_queue = clearml_queue
        set_seed(self._config.data["seed"])
        self._provider = pretrained_model_provider_factory.create(config, mixed_precision)

    # --- training -----------------------------------------------------------------

    def train(self) -> None:
        training_args = self._create_training_arguments()
        tokenizer = self._config.get_hf_tokenizer()
        tokenizer.padding_side = "right"

        model = self._provider.create_model_for_training()
        model = self._apply_finetuning_config(model)

        max_seq_length: int = self._config.params["max_seq_length"]
        src_lang = self._config.language(self._config.train_src_iso)
        trg_lang = self._config.language(self._config.train_trg_iso)
        eos_token_id = tokenizer.eos_token_id

        def encode(example: dict) -> dict:
            prompt = self._config.build_prompt_messages(example["src"], src_lang, trg_lang)
            prompt_ids = prompt.apply_prompt_template(tokenizer, add_generation_prompt=True, tokenize=True)
            completion_ids = tokenizer(example["trg"], add_special_tokens=False)["input_ids"] + [eos_token_id]
            input_ids = (prompt_ids + completion_ids)[:max_seq_length]
            labels = ([LABEL_PAD_TOKEN_ID] * len(prompt_ids) + completion_ids)[:max_seq_length]
            return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

        train_dataset = self._load_text_dataset(
            self._config.exp_dir / self._config.train_src_filename(),
            self._config.exp_dir / self._config.train_trg_filename(),
        )
        eval_dataset = self._load_text_dataset(
            self._config.exp_dir / self._config.val_src_filename(),
            self._config.exp_dir / self._config.val_trg_filename(),
        )
        if train_dataset is not None:
            train_dataset = train_dataset.map(encode, remove_columns=train_dataset.column_names)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(encode, remove_columns=eval_dataset.column_names)

        data_collator = DataCollatorForCausalLM(
            tokenizer, pad_to_multiple_of=8 if (training_args.fp16 or training_args.bf16) else None
        )

        trainer = SilCausalTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
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

        last_checkpoint = find_last_checkpoint(Path(training_args.output_dir))
        train_result = trainer.train(resume_from_checkpoint=str(last_checkpoint) if last_checkpoint else None)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset) if train_dataset is not None else 0
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def _apply_finetuning_config(self, model: PreTrainedModel) -> PreTrainedModel:
        if self._config.finetune_method == FULL_FINETUNE_METHOD:
            return model

        from peft import get_peft_model, prepare_model_for_kbit_training

        gradient_checkpointing = self._config.train["gradient_checkpointing"]
        if self._config.uses_quantization:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        elif gradient_checkpointing:
            model.enable_input_require_grads()

        peft_config = self._build_adapter_config(self._config.adapter, use_dora=self._config.uses_dora)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    @staticmethod
    def _build_adapter_config(adapter: dict, use_dora: bool) -> Any:
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=adapter["rank"],
            lora_alpha=adapter["alpha"],
            lora_dropout=adapter["dropout"],
            target_modules=adapter["target_modules"],
            modules_to_save=adapter.get("modules_to_save"),
            use_dora=use_dora,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    def _load_text_dataset(self, src_path: Path, trg_path: Path) -> Optional[Dataset]:
        if not src_path.is_file() or not trg_path.is_file():
            return None
        corpus = TextFileTextCorpus(src_path).align_rows(TextFileTextCorpus(trg_path))
        sources: List[str] = []
        targets: List[str] = []
        for row in corpus:
            sources.append(row.source_text)
            targets.append(row.target_text)
        if len(sources) == 0:
            return None
        return Dataset.from_dict({"src": sources, "trg": targets})

    def _create_training_arguments(self) -> TrainingArguments:
        dtype = self._config.params["torch_dtype"]
        args = collect_training_args(
            self._config.root,
            TRAINING_ARGS_CONFIG_MAPPING,
            {
                "bf16": self._mixed_precision and dtype == "bfloat16",
                "fp16": self._mixed_precision and dtype == "float16",
            },
            self._clearml_queue,
        )
        return HfArgumentParser(TrainingArguments).parse_dict(args)[0]

    def save_effective_config(self, path: Path) -> None:
        write_effective_config(path, self._config.root, self._create_training_arguments(), TRAINING_ARGS_CONFIG_MAPPING)

    # --- inference ----------------------------------------------------------------

    def _create_inference_model(self, ckpt: Union[CheckpointType, str, int]) -> PreTrainedModel:
        if self._config.model_dir.exists():
            checkpoint_path, _ = self.get_checkpoint_path(ckpt)
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
        src_lang = self._config.language(src_iso)
        trg_lang = self._config.language(trg_iso)
        model = self._get_inference_model(ckpt, src_lang.name, trg_lang.name)
        tokenizer = self._config.get_hf_tokenizer()
        yield from self._generate(model, tokenizer, sentences, src_lang, trg_lang, produce_multiple_translations, False)

    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        tokenizer = self._config.get_hf_tokenizer()
        src_iso = self._config.train_src_iso
        trg_iso = self._config.train_trg_iso
        src_lang = self._config.language(src_iso)
        trg_lang = self._config.language(trg_iso)
        model = self._get_inference_model(ckpt, src_lang.name, trg_lang.name)

        for input_path, translation_path in zip(input_paths, translation_paths):
            file_src_iso, file_trg_iso = self._isos_for_test_file(input_path, src_iso, trg_iso)
            file_src_lang = self._config.language(file_src_iso)
            file_trg_lang = self._config.language(file_trg_iso)
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
                    translation_draft_path = translation_path.with_suffix(f".{draft_index}{translation_path.suffix}")
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

        infer = self._config.infer
        gen_kwargs = build_generation_kwargs(infer, num_return_sequences, tokenizer.pad_token_id)

        device = model.device
        for batch in batch_sentences(sentences, infer["infer_batch_size"]):
            prompts = [
                self._config.build_prompt_messages(sentence, src_lang, trg_lang).apply_prompt_template(
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
