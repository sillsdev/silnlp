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
built with the model's native chat template from ``train.prompt`` and ``infer.prompt``.
"""

import logging
import re
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple, Union

from transformers import PreTrainedModel, PreTrainedTokenizerBase, set_seed

from ..common.confidence_files import generate_confidence_files
from ..common.environment import SilNlpEnv
from ..common.translation_data_structures import DraftGroup, SentenceTranslationGroup
from ..common.utils import merge_dict
from .causal_lm_generator import CausalLMGenerator
from .causal_lm_provider import CausalLMModelSettings, CausalLMProviderFactory, FileCausalLMProviderFactory
from .causal_lm_tokenizer import CausalLMTokenizer
from .causal_lm_training_data import CausalLMTrainingDataSets, InstructionMix, OverlengthRowFilter
from .causal_lm_training_run import CausalLMTrainingArguments, CausalLMTrainingRun
from .checkpoints import CheckpointDirectory, CheckpointType
from .config import InferenceModelParams, NMTModel
from .config_keys import DeprecatedAdapterKey, RenamedConfigKeys
from .example_retrieval import ExamplePool, ExamplePools, ExampleRetrieverFactory
from .experiment_data_set_writer import DataSetWriter
from .experiment_languages import ExperimentLanguages
from .experiment_settings import EvaluationSettings, TrainerSettings
from .finetune_method import FinetuneMethod
from .finetuning import AdapterSettings, Finetuning
from .generation_settings import GenerationSettings
from .instruction_data import InstructionData, InstructionDataSettings, InstructionDataSetWriter
from .llm_config import (
    LLMConfig,
    PromptBuilder,
    PromptConfig,
    PromptDefaults,
    PromptMessagesFactory,
    PromptTemplateCollection,
)
from .local_llm_prompts import LocalLLMPromptMessages, LocalLLMPromptMessagesFactory, LocalLLMPrompts
from .model_name import ModelName
from .prediction_files import PredictionFile

LOGGER = logging.getLogger(__name__)


class TrainPromptConfig(PromptConfig):
    """The train.prompt section, which may rotate through templates read from a file."""

    FIXED = "fixed"
    ROTATING = "rotating"
    _VALID_TYPES = (FIXED, ROTATING)

    # Only meaningful for a fixed prompt; a rotating one takes them from its template file.
    _FIXED_ONLY_KEYS = ("system_message", "instruction_template", "example_format")

    def __init__(self, settings: dict, defaults: PromptDefaults) -> None:
        self._type = str(settings["type"]).lower()
        super().__init__(settings, "train.prompt", defaults)
        self._validate()

    def _apply_defaults(self, defaults: PromptDefaults) -> None:
        # A rotating prompt takes these fields from its template file, not from the experiment.
        if not self.rotates_templates():
            super()._apply_defaults(defaults)

    def _validate(self) -> None:
        if self._type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown train.prompt.type '{self._settings['type']}'. "
                f"Valid options: {', '.join(self._VALID_TYPES)}."
            )
        if not self.rotates_templates():
            if not self._is_unset("template_file"):
                raise ValueError('train.prompt.template_file is only valid with train.prompt.type: "rotating".')
            return

        set_keys = [key for key in self._FIXED_ONLY_KEYS if not self._is_unset(key)]
        if set_keys:
            raise ValueError(
                f"train.prompt.{', train.prompt.'.join(set_keys)} "
                f'{"are" if len(set_keys) > 1 else "is"} only valid with train.prompt.type: "fixed"; '
                "a rotating prompt takes them from its template file."
            )
        if self._is_unset("template_file"):
            raise ValueError('train.prompt.type: "rotating" requires train.prompt.template_file.')

    def rotates_templates(self) -> bool:
        return self._type == self.ROTATING

    def get_template_file(self) -> str:
        return str(self._settings["template_file"])


class LocalLLMConfig(LLMConfig[LocalLLMPromptMessages]):
    _INSTRUCTION_DATA_FILE = "instruction.jsonl"

    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        DeprecatedAdapterKey().apply_to(config)
        RenamedConfigKeys.of_llm().warn_about(config)

        super().__init__(exp_dir, config, environment)

        train_prompt = TrainPromptConfig(self.train["prompt"], self.prompt_defaults())
        self._train_example_pool = self._create_example_pool(train_prompt)
        self._train_prompt_builder = self._create_train_prompt_builder(train_prompt)
        self._disable_eval_if_no_val_split()
        self._tokenizer = CausalLMTokenizer(self.model, self.params["trust_remote_code"])
        self._instruction_data = InstructionDataSettings(self.train["instruction_data"], self._environment.mt_dir)

    def _default_config(self, exp_dir: Path) -> dict:
        return merge_dict(
            super()._default_config(exp_dir),
            {
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
                    # None means "unset"; resolve_prompt_defaults() fills these in.
                    "prompt": {
                        "type": TrainPromptConfig.FIXED,
                        "system_message": None,
                        "instruction_template": None,
                        "example_format": None,
                        "num_examples": 0,
                        "example_selection": {"method": ExampleRetrieverFactory.DEFAULT_METHOD, "model": None},
                        "template_file": None,
                    },
                    "instruction_data": {
                        # Each name must have a matching <name>.jsonl under <mt_dir>/instructions
                        # (see scripts/prepare_instruction_data.py).
                        "datasets": [],
                        "size": 100000,
                        "mix_ratio": 0.1,
                    },
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
                    "adapter": {
                        "rank": 16,
                        "alpha": 32,
                        "dropout": 0.05,
                        "target_modules": "all-linear",
                        "modules_to_save": None,
                    },
                },
                "model": "google/gemma-2-2b-it",
            },
        )

    def _create_prompt_builder(
        self, prompt: PromptConfig, pool: Optional[ExamplePool]
    ) -> PromptBuilder[LocalLLMPromptMessages]:
        self._reject_examples_for_translate_gemma(prompt.get_num_examples(), prompt.get_name())
        return super()._create_prompt_builder(prompt, pool)

    def _create_train_prompt_builder(self, prompt: TrainPromptConfig) -> PromptBuilder[LocalLLMPromptMessages]:
        if not prompt.rotates_templates():
            return self._create_prompt_builder(prompt, self._train_example_pool)

        template_file = prompt.get_template_file()
        num_examples = prompt.get_num_examples()
        self._reject_examples_for_translate_gemma(num_examples, prompt.get_name())
        templates = PromptTemplateCollection.from_file(self._resolve_template_file(template_file))
        templates.validate_for_icl(num_examples, f"{template_file} instruction_template")
        return PromptBuilder(templates, num_examples, self._train_example_pool, self.create_messages_factory())

    def create_messages_factory(self) -> PromptMessagesFactory[LocalLLMPromptMessages]:
        return LocalLLMPromptMessagesFactory()

    def _resolve_template_file(self, template_file: str) -> Path:
        """Resolve against the experiment directory first, so an experiment can ship its own
        templates, then against the MT directory, so several can share one file."""
        for candidate in (self.exp_dir / template_file, self._environment.mt_dir / template_file):
            if candidate.is_file():
                return candidate
        return Path(template_file)

    def has_fixed_prompt(self) -> bool:
        return ModelName(self.model).uses_translate_gemma_template()

    def _reject_examples_for_translate_gemma(self, num_examples: int, name: str) -> None:
        if num_examples > 0 and self.has_fixed_prompt():
            raise RuntimeError(
                "TranslateGemma models do not support few-shot examples in the prompt. "
                f"Set {name}.num_examples to 0 or use a different model."
            )

    def create_data_set_writer(self, force_align: bool) -> DataSetWriter:
        dataset_paths = self._instruction_data.paths()
        return InstructionDataSetWriter(
            super().create_data_set_writer(force_align),
            InstructionData(
                self.files,
                self.exp_dir / self._INSTRUCTION_DATA_FILE,
                dataset_paths,
                self._instruction_data.size() if len(dataset_paths) > 0 else 0,
            ),
        )

    def create_model(
        self,
        mixed_precision: bool = True,
        num_devices: int = 1,
        clearml_queue: Optional[str] = None,
        pretrained_model_provider_factory: Optional[CausalLMProviderFactory] = None,
    ) -> NMTModel:
        if pretrained_model_provider_factory is None:
            pretrained_model_provider_factory = FileCausalLMProviderFactory()
        finetune_method = FinetuneMethod(self.params["finetune_method"])
        trainer_settings = TrainerSettings(self.train)
        model_settings = CausalLMModelSettings(self.params)
        provider = pretrained_model_provider_factory.create(
            self.model, model_settings, finetune_method, trainer_settings, mixed_precision
        )
        languages = self.create_languages()
        example_pools = ExamplePools(self.example_pools())
        return LocalLLMModel(
            CheckpointDirectory(self.model_dir),
            self.infer.get("num_drafts", 1),
            languages,
            CausalLMGenerator(
                provider,
                self._tokenizer,
                self.create_inference_prompts(),
                GenerationSettings(self.infer),
                example_pools,
            ),
            CausalLMTrainingRun(
                provider,
                self._tokenizer,
                CausalLMTrainingDataSets(
                    self.files,
                    self.create_training_prompts(),
                    languages,
                    OverlengthRowFilter(self.params["max_seq_length"]),
                    InstructionMix(
                        self.exp_dir / self._INSTRUCTION_DATA_FILE,
                        self._instruction_data,
                        num_devices,
                        self.data["seed"],
                    ),
                ),
                Finetuning(finetune_method, AdapterSettings(self.params["adapter"]), trainer_settings),
                EvaluationSettings(self.eval),
                trainer_settings,
                CausalLMTrainingArguments(self.root, model_settings, mixed_precision, clearml_queue),
                example_pools,
            ),
            self.data["seed"],
        )

    def create_training_prompts(self) -> LocalLLMPrompts:
        return LocalLLMPrompts(ModelName(self.model), self._train_prompt_builder)

    def create_inference_prompts(self) -> LocalLLMPrompts:
        return LocalLLMPrompts(ModelName(self.model), self._infer_prompt_builder)

    def example_pools(self) -> List[ExamplePool]:
        if self._train_example_pool is None:
            return super().example_pools()
        return [*super().example_pools(), self._train_example_pool]


class LocalLLMModel(NMTModel):
    def __init__(
        self,
        checkpoints: CheckpointDirectory,
        num_drafts: int,
        languages: ExperimentLanguages,
        generator: CausalLMGenerator,
        training_run: CausalLMTrainingRun,
        seed: int,
    ) -> None:
        super().__init__(checkpoints, num_drafts)
        self._languages = languages
        self._generator = generator
        self._training_run = training_run
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
        return self._generator.load(checkpoint_path)

    def load_for_inference(
        self, ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Loads the model and tokenizer for a checkpoint without any translation-specific prompt building."""
        return self._create_inference_model(ckpt), self._generator.loaded_tokenizer()

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
        self._generator.ensure_examples_available()
        src_lang = self._languages.of(src_iso)
        trg_lang = self._languages.of(trg_iso)
        model = self._get_inference_model(ckpt, src_lang.name, trg_lang.name)
        yield from self._generator.generate(
            model, sentences, src_lang, trg_lang, self._num_return_sequences(produce_multiple_translations), False
        )

    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        self._generator.ensure_examples_available()
        src_iso = self._languages.training_source_iso()
        trg_iso = self._languages.training_target_iso()
        src_lang = self._languages.of(src_iso)
        trg_lang = self._languages.of(trg_iso)
        model = self._get_inference_model(ckpt, src_lang.name, trg_lang.name)

        for input_path, translation_path in zip(input_paths, translation_paths):
            file_src_iso, file_trg_iso = self._isos_for_test_file(input_path, src_iso, trg_iso)
            with open(input_path, "r", encoding="utf-8-sig") as src_file:
                sentences = [line.strip() for line in src_file]
            sentence_translation_groups = list(
                self._generator.generate(
                    model,
                    sentences,
                    self._languages.of(file_src_iso),
                    self._languages.of(file_trg_iso),
                    self._num_return_sequences(produce_multiple_translations),
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

    def _num_return_sequences(self, produce_multiple_translations: bool) -> int:
        num_drafts = self.get_num_drafts()
        return num_drafts if (produce_multiple_translations and num_drafts > 1) else 1
