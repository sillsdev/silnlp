"""Shared configuration for LLM translation experiments, whether the model runs locally
(:mod:`silnlp.nmt.local_llm_config`) or is hosted behind an API
(:mod:`silnlp.nmt.remote_llm_config`)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..common.environment import SilNlpEnv
from ..common.utils import merge_dict
from .config import Config, Language
from .corpora import DataFile
from .example_retrieval import (
    TFIDF_METHOD,
    VALID_SELECTION_METHODS,
    Example,
    ExampleFormatter,
    ExamplePool,
    create_example_formatter,
)
from .tokenizer import NullTokenizer, Tokenizer

LOGGER = logging.getLogger(__name__)

EXAMPLES_PLACEHOLDER = "{examples}"


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


@dataclass(frozen=True)
class PromptTemplate:
    system_message: str
    instruction_template: str
    formatter: ExampleFormatter

    @property
    def has_examples_placeholder(self) -> bool:
        return EXAMPLES_PLACEHOLDER in self.instruction_template


class PromptBuilder:
    """Builds the messages for one translation from a fixed or rotating set of prompt templates."""

    messages_class = PromptMessages

    def __init__(self, templates: Sequence[PromptTemplate], num_examples: int, pool: Optional[ExamplePool]) -> None:
        if len(templates) == 0:
            raise ValueError("A prompt needs at least one template.")
        self._templates = list(templates)
        self._num_examples = num_examples
        self._pool = pool

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def templates(self) -> List[PromptTemplate]:
        return self._templates

    @property
    def pool(self) -> Optional[ExamplePool]:
        return self._pool

    def template_for(self, rotation_index: Optional[int]) -> PromptTemplate:
        # Keyed off the row index so a re-run, and each evaluation, renders the same prompts.
        if rotation_index is None or len(self._templates) == 1:
            return self._templates[0]
        return self._templates[rotation_index % len(self._templates)]

    def covers_whole_pool(self) -> bool:
        return self._pool is not None and self._pool.covers_whole_pool(self._num_examples)

    def select_examples(self, query: str, pool_index: Optional[int] = None) -> List[Example]:
        if self._num_examples <= 0 or self._pool is None:
            return []
        return self._pool.select(query, self._num_examples, pool_index)

    def render_examples(
        self, examples: Sequence[Example], src_lang: Language, trg_lang: Language, rotation_index: Optional[int] = None
    ) -> str:
        return self.template_for(rotation_index).formatter.format(examples, src_lang.name, trg_lang.name)

    def build(
        self,
        source: str,
        src_lang: Language,
        trg_lang: Language,
        target: Optional[str] = None,
        pool_index: Optional[int] = None,
        rotation_index: Optional[int] = None,
        **instruction_fields: Any,
    ) -> PromptMessages:
        # Separate indices so rows outside the pool, which get no pool_index, still rotate.
        if rotation_index is None:
            rotation_index = pool_index
        template = self.template_for(rotation_index)
        examples = self.select_examples(source, pool_index)
        instruction = template.instruction_template.format(
            src_lang=src_lang.name,
            trg_lang=trg_lang.name,
            source=source,
            examples=self.render_examples(examples, src_lang, trg_lang, rotation_index),
            **instruction_fields,
        )
        system_message = template.system_message.format(src_lang=src_lang.name, trg_lang=trg_lang.name)
        return self.messages_class(system_message, instruction, target)


def warn_about_examples_placeholder(templates: Sequence[PromptTemplate], num_examples: int, source: str) -> None:
    for i, template in enumerate(templates):
        where = f"{source}[{i}]" if len(templates) > 1 else source
        if num_examples > 0 and not template.has_examples_placeholder:
            LOGGER.warning(
                "num_examples is %d but %s has no '%s', so the retrieved examples are silently discarded.",
                num_examples,
                where,
                EXAMPLES_PLACEHOLDER,
            )
        elif num_examples == 0 and template.has_examples_placeholder:
            LOGGER.warning(
                "num_examples is 0 but %s has an '%s', which always renders as nothing.",
                where,
                EXAMPLES_PLACEHOLDER,
            )


def read_prompt_template_file(path: Path, defaults: dict) -> List[PromptTemplate]:
    """One {system_message, instruction_template, example_format} object per line."""
    if not path.is_file():
        raise RuntimeError(f"The prompt template file {path} does not exist.")
    templates: List[PromptTemplate] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip() == "":
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path} line {line_number} is not valid JSON: {e}") from e
            if not isinstance(entry, dict):
                raise RuntimeError(f"{path} line {line_number} must be a JSON object.")
            unknown = set(entry) - {"system_message", "instruction_template", "example_format"}
            if unknown:
                raise RuntimeError(
                    f"{path} line {line_number} has unknown field(s) {', '.join(sorted(unknown))}. "
                    "Valid fields: system_message, instruction_template, example_format."
                )
            templates.append(
                PromptTemplate(
                    system_message=entry.get("system_message", defaults["system_message"]),
                    instruction_template=entry.get("instruction_template", defaults["instruction_template"]),
                    formatter=create_example_formatter(entry.get("example_format", defaults["example_format"])),
                )
            )
    if len(templates) == 0:
        raise RuntimeError(f"The prompt template file {path} has no templates.")
    return templates


def parse_example_selection(prompt: dict) -> Tuple[str, Optional[str]]:
    selection = prompt["example_selection"]
    # merge_dict() replaces rather than merges when a bare-string override lands on a dict default.
    if isinstance(selection, str):
        selection = {"method": selection}
    method = str(selection["method"]).lower()
    if method not in VALID_SELECTION_METHODS:
        raise ValueError(
            f"Unknown example_selection.method '{method}'. Valid options: {', '.join(VALID_SELECTION_METHODS)}."
        )
    return method, selection.get("model")


def parse_num_examples(prompt: dict, name: str) -> int:
    num_examples = int(prompt["num_examples"])
    if num_examples < 0:
        raise ValueError(f"{name}.num_examples must be non-negative, got {num_examples}.")
    return num_examples


class LLMConfig(Config):
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
    PROMPT_BUILDER_CLASS = PromptBuilder

    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        config = merge_dict(self._default_config(exp_dir), config)
        self._resolve_infer_prompt_defaults(config["infer"]["prompt"])

        super().__init__(exp_dir, config, environment)

        if len(self.src_isos) > 1 or len(self.trg_isos) > 1:
            raise RuntimeError(
                f"{type(self).__name__} experiments only support a single source language and a single "
                "target language."
            )
        self._infer_prompt_builder = self._create_prompt_builder(self.infer["prompt"], "infer.prompt")

    def _resolve_infer_prompt_defaults(self, prompt: dict) -> None:
        resolve_prompt_defaults(prompt, type(self))

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
                    "example_selection": {"method": TFIDF_METHOD, "model": None},
                },
            },
        }

    def _create_prompt_builder(self, prompt: dict, name: str) -> PromptBuilder:
        num_examples = parse_num_examples(prompt, name)
        templates = [
            PromptTemplate(
                system_message=prompt["system_message"],
                instruction_template=prompt["instruction_template"],
                formatter=create_example_formatter(prompt["example_format"]),
            )
        ]
        warn_about_examples_placeholder(templates, num_examples, f"{name}.instruction_template")
        return self.PROMPT_BUILDER_CLASS(templates, num_examples, self._create_example_pool(prompt, num_examples))

    def _create_example_pool(self, prompt: dict, num_examples: int) -> Optional[ExamplePool]:
        if num_examples <= 0:
            return None
        method, model_name = parse_example_selection(prompt)
        return ExamplePool(self._example_corpus_paths(), method, model_name)

    def _example_corpus_paths(self) -> List[Tuple[Path, Path]]:
        return [(self.exp_dir / self.train_src_filename(), self.exp_dir / self.train_trg_filename())]

    @property
    def infer_prompt_builder(self) -> PromptBuilder:
        return self._infer_prompt_builder

    def prompt_builders(self) -> List[PromptBuilder]:
        return [self._infer_prompt_builder]

    def check_example_corpora(self) -> None:
        for builder in self.prompt_builders():
            if builder.pool is not None:
                builder.pool.ensure_available()

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

    def create_tokenizer(self) -> Tokenizer:
        # Data prep and test.py only tokenize and detokenize with this; both are raw text here.
        return NullTokenizer()

    def _build_vocabs(self, stats: bool = False) -> None:
        # LLMs come with their own vocabulary; there is no SentencePiece model to build.
        return

    def _write_dictionary(
        self,
        tokenizer: Tokenizer,
        src_terms_files: List[Tuple[DataFile, List[str]]],
        trg_terms_files: List[Tuple[DataFile, List[str]]],
    ) -> int:
        return 0


def resolve_prompt_defaults(prompt: dict, config_class: type) -> None:
    """The default instruction template depends on num_examples, so a zero-shot prompt has no
    dangling examples placeholder."""
    if prompt.get("system_message") is None:
        prompt["system_message"] = config_class.DEFAULT_SYSTEM_MESSAGE
    if prompt.get("example_format") is None:
        prompt["example_format"] = config_class.DEFAULT_EXAMPLE_FORMAT
    if prompt.get("instruction_template") is None:
        prompt["instruction_template"] = (
            config_class.DEFAULT_FEW_SHOT_INSTRUCTION_TEMPLATE
            if int(prompt.get("num_examples", 0)) > 0
            else config_class.DEFAULT_INSTRUCTION_TEMPLATE
        )
