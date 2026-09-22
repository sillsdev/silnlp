"""Shared configuration for LLM translation experiments, whether the model runs locally
(:mod:`silnlp.nmt.local_llm_config`) or is hosted behind an API
(:mod:`silnlp.nmt.remote_llm_config`)."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

from ..common.environment import SilNlpEnv
from ..common.utils import merge_dict
from .config import Config, Language
from .corpora import DataFile
from .example_retrieval import (
    CorpusPair,
    CorpusPairProvider,
    Example,
    ExampleFormatter,
    ExampleFormatterFactory,
    ExamplePool,
    ExamplePoolSummary,
    ExampleRetriever,
    ExampleRetrieverFactory,
    FixedCorpusPairProvider,
)
from .tokenizer import NullTokenizer, Tokenizer

LOGGER = logging.getLogger(__name__)


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

    def with_additional_context(self, context: str) -> "PromptMessages":
        """Context goes first, so the prompt prefix a provider can cache does not vary per request."""
        system_message = f"{self.system_message}\n\n{context}" if self.system_message else context
        return PromptMessages(system_message, self.instruction, self.target)


TPromptMessages = TypeVar("TPromptMessages", bound=PromptMessages)


class PromptMessagesFactory(ABC, Generic[TPromptMessages]):
    """Creates the PromptMessages a model needs, so PromptBuilder need not know which that is."""

    @abstractmethod
    def create(self, system_message: str, instruction: str, target: Optional[str]) -> TPromptMessages: ...


class PlainPromptMessagesFactory(PromptMessagesFactory[PromptMessages]):
    def create(self, system_message: str, instruction: str, target: Optional[str]) -> PromptMessages:
        return PromptMessages(system_message, instruction, target)


class MalformedPromptTemplateException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self._message = message

    def get_message(self) -> str:
        return self._message


@dataclass(frozen=True)
class PromptTemplate:
    _EXAMPLES_PLACEHOLDER = "{examples}"
    _REQUIRED_FIELDS = ("system_message", "instruction_template", "example_format")

    system_message: str
    instruction_template: str
    formatter: ExampleFormatter

    def _has_examples_placeholder(self) -> bool:
        return self._EXAMPLES_PLACEHOLDER in self.instruction_template

    def describe_examples_mismatch(self, num_examples: int) -> Optional[str]:
        """A description to appear in the error message"""
        if num_examples > 0 and not self._has_examples_placeholder():
            return f"has no '{self._EXAMPLES_PLACEHOLDER}', so the retrieved examples are silently discarded"
        if num_examples == 0 and self._has_examples_placeholder():
            return f"has an '{self._EXAMPLES_PLACEHOLDER}', which always renders as nothing"
        return None

    def to_prompt_message(
        self,
        source_text: str,
        target_text: Optional[str],
        src_lang: Language,
        trg_lang: Language,
        examples: Sequence[Example],
        messages_factory: PromptMessagesFactory[TPromptMessages],
        num_segments: int = 1,
    ) -> TPromptMessages:
        examples_str = self.render_examples(examples, src_lang, trg_lang)
        formatted_instruction = self.instruction_template.format(
            src_lang=src_lang.name,
            trg_lang=trg_lang.name,
            source=source_text,
            examples=examples_str,
            num_segments=num_segments,
        )
        formatted_system_message = self.system_message.format(src_lang=src_lang.name, trg_lang=trg_lang.name)
        return messages_factory.create(formatted_system_message, formatted_instruction, target_text)

    def render_examples(self, examples: Sequence[Example], src_lang: Language, trg_lang: Language) -> str:
        return self.formatter.format(examples, src_lang.name, trg_lang.name)

    @classmethod
    def from_json(cls, json_str: str) -> "PromptTemplate":
        entry = cls._parse_json_object(json_str)
        try:
            formatter = ExampleFormatterFactory.create(entry["example_format"])
        except ValueError as e:
            raise MalformedPromptTemplateException(f"Prompt template {json_str} has an invalid example_format: {e}")
        return PromptTemplate(entry["system_message"], entry["instruction_template"], formatter)

    @classmethod
    def _parse_json_object(cls, json_str: str) -> dict:
        try:
            entry = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise MalformedPromptTemplateException(f"Prompt template {json_str} is not valid JSON: {e}") from e
        if not isinstance(entry, dict):
            raise MalformedPromptTemplateException(f"Prompt template {json_str} must be a JSON object.")
        unknown_fields = set(entry) - set(cls._REQUIRED_FIELDS)
        if unknown_fields:
            raise MalformedPromptTemplateException(
                f"Prompt template {json_str} has unknown field(s) {', '.join(sorted(unknown_fields))}. "
                f"Valid fields: {', '.join(cls._REQUIRED_FIELDS)}."
            )
        missing = [field for field in cls._REQUIRED_FIELDS if field not in entry]
        if missing:
            raise MalformedPromptTemplateException(
                f"Prompt template {json_str} is missing required field(s) {', '.join(missing)}."
            )
        return entry


class PromptTemplateCollection:
    def __init__(self, prompt_templates: List[PromptTemplate]) -> None:
        self._templates = prompt_templates

    def __len__(self) -> int:
        return len(self._templates)

    def template_for(self, rotation_index: Optional[int]) -> PromptTemplate:
        if rotation_index is None:
            return self._templates[0]
        return self._templates[rotation_index % len(self)]

    def validate_for_icl(self, num_examples: int, source: str) -> None:
        for i, template in enumerate(self._templates):
            mismatch = template.describe_examples_mismatch(num_examples)
            if mismatch is not None:
                where = f"{source}[{i}]" if len(self) > 1 else source
                LOGGER.warning("num_examples is %d but %s %s.", num_examples, where, mismatch)

    @classmethod
    def from_fixed_prompt_template(cls, prompt_template: PromptTemplate) -> "PromptTemplateCollection":
        return cls([prompt_template])

    @classmethod
    def from_file(cls, prompt_file_path: Path) -> "PromptTemplateCollection":
        if not prompt_file_path.is_file():
            raise FileNotFoundError(f"The prompt template file {prompt_file_path} does not exist.")
        templates: List[PromptTemplate] = []
        with prompt_file_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip() == "":
                    continue
                try:
                    templates.append(PromptTemplate.from_json(line))
                except MalformedPromptTemplateException as e:
                    LOGGER.warning(e.get_message())
        if len(templates) == 0:
            raise RuntimeError(f"The prompt template file {prompt_file_path} has no templates.")

        return cls(templates)


class PromptBuilder(Generic[TPromptMessages]):
    """Builds the messages for one translation from a fixed or rotating set of prompt templates."""

    def __init__(
        self,
        templates: PromptTemplateCollection,
        num_examples: int,
        pool: Optional[ExamplePool],
        messages_factory: PromptMessagesFactory[TPromptMessages],
    ) -> None:
        if len(templates) == 0:
            raise ValueError("No valid prompt templates were supplied.")
        self._templates = templates
        self._num_examples = num_examples
        self._pool = pool
        self._messages_factory = messages_factory

    def get_num_examples(self) -> int:
        return self._num_examples

    def covers_whole_pool(self) -> bool:
        return self._pool is not None and self._pool.covers_whole_pool(self._num_examples)

    def select_examples(self, query: str, pool_index: Optional[int] = None) -> List[Example]:
        if self._num_examples <= 0 or self._pool is None:
            return []
        return self._pool.select(query, self._num_examples, pool_index)

    def build(
        self,
        source: str,
        src_lang: Language,
        trg_lang: Language,
        target: Optional[str] = None,
        pool_index: Optional[int] = None,
        rotation_index: Optional[int] = None,
        examples: Optional[Sequence[Example]] = None,
    ) -> TPromptMessages:
        return self._build(
            source,
            src_lang,
            trg_lang,
            target=target,
            pool_index=pool_index,
            rotation_index=rotation_index,
            examples=examples,
        )

    def _build(
        self,
        source: str,
        src_lang: Language,
        trg_lang: Language,
        num_segments: int = 1,
        target: Optional[str] = None,
        pool_index: Optional[int] = None,
        rotation_index: Optional[int] = None,
        examples: Optional[Sequence[Example]] = None,
    ) -> TPromptMessages:
        # Separate indices so rows outside the pool, which get no pool_index, still rotate.
        if rotation_index is None:
            rotation_index = pool_index
        template = self._templates.template_for(rotation_index)
        if examples is None:
            examples = self.select_examples(source, pool_index)
        return template.to_prompt_message(
            source, target, src_lang, trg_lang, examples, self._messages_factory, num_segments
        )

    def render_pool(self, src_lang: Language, trg_lang: Language) -> str:
        if self._pool is None:
            return ""
        return self._templates.template_for(None).render_examples(self._pool.all_examples(), src_lang, trg_lang)


@dataclass(frozen=True)
class PromptDefaults:
    system_message: str
    instruction_template: str
    few_shot_instruction_template: str
    example_format: Union[str, dict]

    def instruction_template_for(self, num_examples: int) -> str:
        return self.few_shot_instruction_template if num_examples > 0 else self.instruction_template


class PromptConfig:
    """A prompt section of an experiment config, e.g., infer.prompt."""

    def __init__(self, settings: dict, name: str, defaults: PromptDefaults) -> None:
        self._settings = settings
        self._name = name
        self._apply_defaults(defaults)

    def get_name(self) -> str:
        return self._name

    def get_num_examples(self) -> int:
        num_examples = int(self._settings["num_examples"])
        if num_examples < 0:
            raise ValueError(f"{self._name}.num_examples must be non-negative, got {num_examples}.")
        return num_examples

    def create_retriever(self) -> ExampleRetriever:
        selection = self._settings["example_selection"]
        # merge_dict() replaces rather than merges when a bare-string override lands on a dict default.
        if isinstance(selection, str):
            selection = {"method": selection}
        return ExampleRetrieverFactory.create(str(selection["method"]), selection.get("model"))

    def create_template(self, instruction_template: Optional[str] = None) -> PromptTemplate:
        return PromptTemplate(
            system_message=self._settings["system_message"],
            instruction_template=instruction_template or self._settings["instruction_template"],
            formatter=ExampleFormatterFactory.create(self._settings["example_format"]),
        )

    def _is_unset(self, key: str) -> bool:
        return self._settings.get(key) is None

    def _apply_defaults(self, defaults: PromptDefaults) -> None:
        if self._is_unset("system_message"):
            self._settings["system_message"] = defaults.system_message
        if self._is_unset("example_format"):
            self._settings["example_format"] = defaults.example_format
        if self._is_unset("instruction_template"):
            self._settings["instruction_template"] = defaults.instruction_template_for(
                int(self._settings.get("num_examples", 0))
            )


class LLMConfig(Config, Generic[TPromptMessages]):
    """An experiment that translates by prompting an LLM, fine-tuned locally or hosted."""

    DEFAULT_SYSTEM_MESSAGE = ""
    DEFAULT_INSTRUCTION_TEMPLATE = "Translate the following text from {src_lang} to {trg_lang}.\n\n{source}"
    DEFAULT_FEW_SHOT_INSTRUCTION_TEMPLATE = (
        "Translate the following text from {src_lang} to {trg_lang}.\n\n"
        "Here are example translations to follow for style, terminology, and spelling:\n\n"
        "{examples}"
        "Now translate this text:\n\n{source}"
    )
    DEFAULT_EXAMPLE_FORMAT: Union[str, dict] = "text"

    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        config = merge_dict(self._default_config(exp_dir), config)
        infer_prompt = self._create_infer_prompt_config(config["infer"]["prompt"])

        super().__init__(exp_dir, config, environment)

        if len(self.src_isos) > 1 or len(self.trg_isos) > 1:
            raise RuntimeError(
                f"{type(self).__name__} experiments only support a single source language and a single "
                "target language."
            )
        self._infer_example_pool = self._create_example_pool(infer_prompt)
        self._infer_prompt_builder = self._create_prompt_builder(infer_prompt, self._infer_example_pool)

    def prompt_defaults(self) -> PromptDefaults:
        return PromptDefaults(
            self.DEFAULT_SYSTEM_MESSAGE,
            self.DEFAULT_INSTRUCTION_TEMPLATE,
            self.DEFAULT_FEW_SHOT_INSTRUCTION_TEMPLATE,
            self.DEFAULT_EXAMPLE_FORMAT,
        )

    def _create_infer_prompt_config(self, settings: dict) -> PromptConfig:
        return PromptConfig(settings, "infer.prompt", self.prompt_defaults())

    def _default_config(self, exp_dir: Path) -> dict:
        return {
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
            "infer": {
                # None means "unset"; resolve_prompt_defaults() fills these in.
                "prompt": {
                    "system_message": None,
                    "instruction_template": None,
                    "example_format": None,
                    "num_examples": 0,
                    "example_selection": {"method": ExampleRetrieverFactory.DEFAULT_METHOD, "model": None},
                },
            },
        }

    def _create_prompt_builder(
        self, prompt: PromptConfig, pool: Optional[ExamplePool]
    ) -> PromptBuilder[TPromptMessages]:
        num_examples = prompt.get_num_examples()
        templates = PromptTemplateCollection.from_fixed_prompt_template(prompt.create_template())
        templates.validate_for_icl(num_examples, f"{prompt.get_name()}.instruction_template")
        return PromptBuilder(templates, num_examples, pool, self.create_messages_factory())

    @abstractmethod
    def create_messages_factory(self) -> PromptMessagesFactory[TPromptMessages]: ...

    def _create_example_pool(self, prompt: PromptConfig) -> Optional[ExamplePool]:
        if prompt.get_num_examples() <= 0:
            return None
        return ExamplePool(self._create_corpus_pair_provider(), prompt.create_retriever())

    def _create_corpus_pair_provider(self) -> CorpusPairProvider:
        return FixedCorpusPairProvider(self._train_corpus_pair())

    def _train_corpus_pair(self) -> CorpusPair:
        return CorpusPair(self.exp_dir / self.train_src_filename(), self.exp_dir / self.train_trg_filename())

    def get_infer_prompt_builder(self) -> PromptBuilder[TPromptMessages]:
        return self._infer_prompt_builder

    def example_pools(self) -> List[ExamplePool]:
        return [pool for pool in (self._infer_example_pool,) if pool is not None]

    def check_example_corpora(self) -> None:
        for pool in self.example_pools():
            pool.ensure_available()

    def save_example_index(self, directory: Path) -> None:
        if self._infer_example_pool is not None:
            self._infer_example_pool.save_index(directory)

    def load_example_index(self, directory: Path) -> bool:
        return self._infer_example_pool is not None and self._infer_example_pool.load_index(directory)

    def summarize_example_pool(self) -> Optional[ExamplePoolSummary]:
        return None if self._infer_example_pool is None else self._infer_example_pool.summarize()

    def lang_name(self, iso: str) -> str:
        return self.data["lang_codes"].get(iso, iso)

    def language(self, iso: str) -> Language:
        return Language(iso=iso, name=self.lang_name(iso))

    def get_train_src_iso(self) -> str:
        return self.default_test_src_iso or (next(iter(self.src_isos)) if len(self.src_isos) > 0 else "")

    def get_train_trg_iso(self) -> str:
        return self.default_test_trg_iso or (next(iter(self.trg_isos)) if len(self.trg_isos) > 0 else "")

    # We don't need to build a SentencePiece tokenizer for LLMs -- the vocabulary is built in
    def create_tokenizer(self) -> Tokenizer:
        return NullTokenizer()

    def _build_vocabs(self, stats: bool = False) -> None:
        return

    def _write_dictionary(
        self,
        tokenizer: Tokenizer,
        src_terms_files: List[Tuple[DataFile, List[str]]],
        trg_terms_files: List[Tuple[DataFile, List[str]]],
    ) -> int:
        return 0
