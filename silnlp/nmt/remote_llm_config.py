"""In-context learning translation with a hosted LLM: a Config/NMTModel implementation that
prompts with examples from the training corpus instead of fine-tuning, via LiteLLM."""

import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

import yaml

from ..common.environment import SilNlpEnv
from ..common.translation_data_structures import DraftGroup, SentenceTranslation, SentenceTranslationGroup
from ..common.translator import generate_confidence_files
from ..common.utils import merge_dict
from .config import CheckpointType, Language, NMTModel
from .example_retrieval import (
    CorpusPair,
    CorpusPairProvider,
    Example,
    ExampleRetrieverFactory,
    PreferredCorpusPairProvider,
)
from .llm_config import (
    LLMConfig,
    PlainPromptMessagesFactory,
    PromptBuilder,
    PromptConfig,
    PromptDefaults,
    PromptMessages,
    PromptMessagesFactory,
    PromptTemplateCollection,
)

LOGGER = logging.getLogger(__name__)


class ModelReply:
    """One reply from the model, which a batched request asks for as one numbered line per segment."""

    _CODE_FENCE = re.compile(r"^\s*```[^\n]*\n(.*?)\n?\s*```\s*$", re.DOTALL)
    _NUMBERED_LINE = re.compile(r"^\s*(\d{1,4})\s*[.):\]]\s*(.*)$")

    def __init__(self, text: str) -> None:
        self._text = text

    def strip_code_fence(self) -> str:
        match = self._CODE_FENCE.match(self._text.strip())
        return match.group(1) if match is not None else self._text

    def parse(self, num_segments: int) -> Optional[List[str]]:
        """None when the reply is malformed, which is the signal for the caller's recovery ladder.
        Unnumbered lines continue the preceding translation, and any preamble is ignored."""
        if num_segments <= 0:
            return []
        parsed: Dict[int, List[str]] = {}
        current: Optional[int] = None
        for line in self.strip_code_fence().splitlines():
            match = self._NUMBERED_LINE.match(line)
            if match is not None:
                index = int(match.group(1))
                if index in parsed:
                    return None
                current = index
                parsed[index] = [match.group(2).strip()]
            elif current is not None and line.strip() != "":
                parsed[current].append(line.strip())
        if set(parsed) != set(range(1, num_segments + 1)):
            return None
        return [" ".join(part for part in parsed[i] if part != "").strip() for i in range(1, num_segments + 1)]


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float


@dataclass(frozen=True)
class Completion:
    """A reply from the model, with whatever the provider reported alongside it."""

    text: str
    token_logprobs: List[TokenLogprob] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # None when LiteLLM has no pricing for the model, which is not the same as free.
    cost: Optional[float] = None

    def to_sentence_translation(self) -> SentenceTranslation:
        """``tokens`` holds the whole translation, not the provider's subword tokens, because the
        predictions file is raw text written by space-joining them."""
        return SentenceTranslation(
            self.text,
            [self.text],
            [entry.logprob for entry in self.token_logprobs],
            self.mean_logprob(),
            starts_with_special_token=False,
        )

    def mean_logprob(self) -> Optional[float]:
        if len(self.token_logprobs) == 0:
            return None
        return sum(entry.logprob for entry in self.token_logprobs) / len(self.token_logprobs)


@dataclass
class UsageTotals:
    """Running totals for one translation run."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    unpriced_requests: int = 0

    def __post_init__(self) -> None:
        # Requests are made from several threads.
        self._lock = threading.Lock()

    def add(self, completion: Completion) -> None:
        with self._lock:
            self.requests += 1
            self.prompt_tokens += completion.prompt_tokens
            self.completion_tokens += completion.completion_tokens
            if completion.cost is None:
                self.unpriced_requests += 1
            else:
                self.cost += completion.cost

    def describe(self) -> str:
        summary = (
            f"{self.requests:,} requests, {self.prompt_tokens:,} prompt + "
            f"{self.completion_tokens:,} completion tokens"
        )
        if self.unpriced_requests == 0:
            return f"{summary}, ${self.cost:.4f}"
        if self.unpriced_requests == self.requests:
            return f"{summary}; cost unavailable (no pricing for this model)"
        return f"{summary}, ${self.cost:.4f} excluding {self.unpriced_requests:,} unpriced requests"


@dataclass(frozen=True)
class CompletionSettings:
    """What a provider needs for one request, apart from the messages themselves."""

    temperature: float
    max_new_tokens: int
    num_retries: int
    request_timeout: int


class LiteLLMResponse:
    def __init__(self, response: Any, litellm: Any) -> None:
        self._response = response
        self._litellm = litellm

    def to_completion(self, want_logprobs: bool) -> Completion:
        return Completion(
            self.text(),
            self.token_logprobs() if want_logprobs else [],
            prompt_tokens=self.prompt_tokens(),
            completion_tokens=self.completion_tokens(),
            cost=self.cost(),
        )

    def text(self) -> str:
        content = self._field(self._field(self._choice(), "message"), "content")
        return content if content is not None else ""

    def token_logprobs(self) -> List[TokenLogprob]:
        """A provider without logprobs omits them rather than failing, so every level is optional."""
        content = self._field(self._field(self._choice(), "logprobs"), "content")
        if not content:
            return []
        token_logprobs: List[TokenLogprob] = []
        for entry in content:
            token = self._field(entry, "token")
            logprob = self._field(entry, "logprob")
            if token is None or logprob is None:
                continue
            token_logprobs.append(TokenLogprob(str(token), float(logprob)))
        return token_logprobs

    def prompt_tokens(self) -> int:
        return int(self._field(self._usage(), "prompt_tokens") or 0)

    def completion_tokens(self) -> int:
        return int(self._field(self._usage(), "completion_tokens") or 0)

    def cost(self) -> Optional[float]:
        try:
            return float(self._litellm.completion_cost(completion_response=self._response))
        except Exception:
            LOGGER.debug("No pricing available for this response; reporting its cost as unknown.", exc_info=True)
            return None

    def _choice(self) -> Any:
        return self._field(self._response, "choices")[0]

    def _usage(self) -> Any:
        return self._field(self._response, "usage")

    def _field(self, obj: Any, name: str) -> Any:
        """A LiteLLM response may be a dict or a pydantic model."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)


class CompletionClient(ABC):
    # Indirected so tests can substitute a scripted client without a network call.

    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion: ...

    def supports_logprobs(self) -> bool:
        return False

    def count_tokens(self, text: str) -> Optional[int]:
        """None means the caller skips its size check rather than run it against a guess."""
        return None


class LiteLLMCompletionClient(CompletionClient):
    def __init__(self, model: str, settings: CompletionSettings, extra_kwargs: Optional[dict] = None) -> None:
        self._model = model
        self._settings = settings
        self._extra_kwargs: Dict[str, Any] = dict(extra_kwargs or {})
        self._litellm = self._import_litellm()

    def _import_litellm(self):
        # Deferred import, because the import is slow and the package is optional.
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "Remote LLM experiments require the 'litellm' package, which is part of the "
                "'llm' extra. Install it with `poetry install -E llm`."
            ) from e
        return litellm

    def count_tokens(self, text: str) -> Optional[int]:
        try:
            return int(self._litellm.token_counter(model=self._model, text=text))
        except Exception:
            LOGGER.warning(
                "Could not count tokens for '%s'; skipping the prompt size check.", self._model, exc_info=True
            )
            return None

    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        extra_kwargs = dict(self._extra_kwargs)
        if logprobs:
            extra_kwargs["logprobs"] = True
        response = self._litellm.completion(
            model=self._model,
            messages=messages,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_new_tokens,
            num_retries=self._settings.num_retries,
            timeout=self._settings.request_timeout,
            **extra_kwargs,
        )
        return LiteLLMResponse(response, self._litellm).to_completion(logprobs)

    def supports_logprobs(self) -> bool:
        try:
            supported = self._litellm.get_supported_openai_params(self._model) or []
        except Exception:
            LOGGER.warning("Could not determine which parameters %s supports.", self._model, exc_info=True)
            return False
        return "logprobs" in supported


class CompletionClientFactory(ABC):
    @abstractmethod
    def create(self, config: "RemoteLLMConfig") -> CompletionClient: ...


class LiteLLMCompletionClientFactory(CompletionClientFactory):
    def create(self, config: "RemoteLLMConfig") -> CompletionClient:
        return LiteLLMCompletionClient(config.model, config.create_completion_settings(), config.get_litellm_options())


class BatchPromptBuilder(PromptBuilder[PromptMessages]):
    """Builds one request carrying several segments, numbered so the reply can be split apart."""

    def build_batch(
        self,
        sources: Sequence[str],
        src_lang: Language,
        trg_lang: Language,
        examples: Optional[Sequence[Example]] = None,
    ) -> PromptMessages:
        numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(sources, 1))
        return self._build(numbered, src_lang, trg_lang, len(sources), examples=examples)


class RemotePromptConfig(PromptConfig):
    """The infer.prompt section of a hosted-model experiment, which also has a batch instruction template."""

    def __init__(
        self,
        settings: dict,
        defaults: PromptDefaults,
        single_default: str,
        batch_default: str,
        few_shot_batch_default: str,
    ) -> None:
        self._batch_default = batch_default
        self._few_shot_batch_default = few_shot_batch_default
        # Captured before the defaults land, since a whole-corpus prompt needs the plain default, not few-shot.
        self._corpus_instruction_template = settings["instruction_template"] or single_default
        self._corpus_batch_instruction_template = settings["batch_instruction_template"] or batch_default
        super().__init__(settings, "infer.prompt", defaults)

    def _apply_defaults(self, defaults: PromptDefaults) -> None:
        super()._apply_defaults(defaults)
        if self._is_unset("batch_instruction_template"):
            self._settings["batch_instruction_template"] = (
                self._few_shot_batch_default if self.get_num_examples() > 0 else self._batch_default
            )

    def get_batch_instruction_template(self) -> str:
        return self._settings["batch_instruction_template"]

    def get_corpus_instruction_template(self) -> str:
        return self._corpus_instruction_template

    def get_corpus_batch_instruction_template(self) -> str:
        return self._corpus_batch_instruction_template


class RemoteLLMConfig(LLMConfig[PromptMessages]):
    _SYSTEM_MESSAGE = (
        "You are a member of a Bible translation team translating from {src_lang} into {trg_lang}. "
        "Your job is to produce the translation this team would produce, not a translation of your "
        "own.\n\n"
        "Any examples you are given are the team's own completed work, and they are your authority. "
        "Study them and follow what they show you about:\n"
        "- Style: how closely the team follows the source wording rather than restructuring it into "
        "natural {trg_lang}, their sentence length and register, and how much implicit information "
        "they make explicit.\n"
        "- Key terms: the rendering the team has settled on for recurring theological terms, and "
        "their spelling of the names of people, places, and peoples. Reuse these exactly; never "
        "substitute a synonym or a variant spelling.\n"
        "- Exegesis: where the source is ambiguous, resolve it the way the team resolved comparable "
        "passages.\n"
        "- Orthography: their spelling conventions, punctuation, and the way they mark direct "
        "speech.\n\n"
        "Follow the examples in preference to any published {trg_lang} translation you may recall. "
        "Where they do not settle a question, make the choice a careful member of this team would "
        "make, and stay consistent with it. Translate what the source says: add nothing it does not "
        "say, and leave out nothing it does.\n\n"
        "Reply with only the translation itself - no commentary, notes, alternatives, explanations, "
        "or verse numbers."
    )

    _SINGLE_INSTRUCTION = (
        "Translate this {src_lang} passage into {trg_lang} as the team would translate it. Reply with "
        "only the translation.\n\n{source}"
    )

    _BATCH_INSTRUCTION = (
        "Translate the following {num_segments} consecutive {src_lang} passages into {trg_lang} as the "
        "team would translate them. Some may be section headings rather than verses. Read them "
        "together, so that participants, pronouns, and the flow of the passage stay consistent across "
        "them, but translate each one on its own.\n"
        "Reply with exactly {num_segments} lines, one per passage, in the same order, each formatted "
        "as `<number>. <translation>`. Do not merge, split, reorder, or omit passages, and do not add "
        "any other text.\n\n{source}"
    )

    _EXAMPLES_HEADING = (
        "The team has already translated these passages. They are your model for this team's style, "
        "terminology, and exegesis:\n\n{examples}"
    )

    _CORPUS_HEADING = (
        "This is everything the team has translated so far. It is your reference for this team's "
        "style, terminology, exegesis, spelling, and punctuation:"
    )

    _FEW_SHOT_BATCH_INSTRUCTION = _EXAMPLES_HEADING + _BATCH_INSTRUCTION

    DEFAULT_SYSTEM_MESSAGE = _SYSTEM_MESSAGE
    DEFAULT_INSTRUCTION_TEMPLATE = _SINGLE_INSTRUCTION
    DEFAULT_FEW_SHOT_INSTRUCTION_TEMPLATE = _EXAMPLES_HEADING + _SINGLE_INSTRUCTION
    DEFAULT_EXAMPLE_FORMAT = {"type": "text", "template": "{src_lang}: {source}\n{trg_lang}: {target}\n\n"}

    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        super().__init__(exp_dir, config, environment)
        self._validate()
        prompt = self._infer_prompt_config()
        self._batch_prompt_builder = self._create_batch_prompt_builder(
            prompt.get_batch_instruction_template(), "infer.prompt.batch_instruction_template"
        )
        self._corpus_prompt_builder = self._create_single_prompt_builder(prompt.get_corpus_instruction_template())
        self._corpus_batch_prompt_builder = self._create_batch_prompt_builder(
            prompt.get_corpus_batch_instruction_template()
        )
        self._disable_eval_if_no_val_split()

    def _default_config(self, exp_dir: Path) -> dict:
        return merge_dict(
            super()._default_config(exp_dir),
            {
                "train": {
                    "output_dir": str(exp_dir / "run"),
                },
                # No training loop, so these exist only because the shared config plumbing reads them.
                "eval": {
                    "eval_strategy": "no",
                    "early_stopping": None,
                    "load_best_model_at_end": False,
                    "metric_for_best_model": None,
                    "greater_is_better": False,
                    "multi_ref_eval": False,
                },
                "infer": {
                    "prompt": {
                        "num_examples": 10,
                        "example_selection": {"method": ExampleRetrieverFactory.DEFAULT_METHOD, "model": None},
                        "batch_instruction_template": None,
                    },
                    "infer_batch_size": 1,
                    "num_drafts": 1,
                    "temperature": 0.2,
                    "max_new_tokens": 4096,
                    "concurrency": 4,
                    "num_retries": 3,
                    "request_timeout": 120,
                    "max_context_tokens": 180000,
                },
                "params": {
                    # Passed straight through to litellm.completion (api_base, extra_headers, ...).
                    "litellm": {},
                },
                "model": "",
            },
        )

    def _create_corpus_pair_provider(self) -> CorpusPairProvider:
        """An experiment preprocessed for a tokenized model has readable text only in the detok files."""
        detokenized = CorpusPair(
            self.exp_dir / self.train_src_detok_filename(),
            self.exp_dir / self.train_trg_detok_filename(),
        )
        return PreferredCorpusPairProvider(detokenized, self._train_corpus_pair())

    def build_corpus_block(self, rendered_examples: str) -> str:
        return f"{self._CORPUS_HEADING}\n\n{rendered_examples}" if rendered_examples else ""

    def _create_infer_prompt_config(self, settings: dict) -> RemotePromptConfig:
        self._infer_prompt = RemotePromptConfig(
            settings,
            self.prompt_defaults(),
            self._SINGLE_INSTRUCTION,
            self._BATCH_INSTRUCTION,
            self._FEW_SHOT_BATCH_INSTRUCTION,
        )
        return self._infer_prompt

    def create_messages_factory(self) -> PromptMessagesFactory[PromptMessages]:
        return PlainPromptMessagesFactory()

    def _create_single_prompt_builder(self, instruction_template: str) -> PromptBuilder[PromptMessages]:
        prompt = self._infer_prompt_config()
        return PromptBuilder(
            self._variant_templates(instruction_template),
            prompt.get_num_examples(),
            self._infer_example_pool,
            self.create_messages_factory(),
        )

    def _create_batch_prompt_builder(
        self, instruction_template: str, validate_as: Optional[str] = None
    ) -> BatchPromptBuilder:
        prompt = self._infer_prompt_config()
        templates = self._variant_templates(instruction_template)
        if validate_as is not None:
            templates.validate_for_icl(prompt.get_num_examples(), validate_as)
        return BatchPromptBuilder(
            templates, prompt.get_num_examples(), self._infer_example_pool, self.create_messages_factory()
        )

    def _infer_prompt_config(self) -> RemotePromptConfig:
        return self._infer_prompt

    def _variant_templates(self, instruction_template: str) -> PromptTemplateCollection:
        """Every variant draws on the single-segment builder's examples, so the corpus is indexed once."""
        return PromptTemplateCollection.from_fixed_prompt_template(
            self._infer_prompt_config().create_template(instruction_template)
        )

    def _validate(self) -> None:
        if not str(self.model).strip():
            raise ValueError(
                "An in-context learning experiment needs a 'model' in LiteLLM format, "
                "e.g. 'anthropic/claude-sonnet-4-5', 'gpt-4o', or 'gemini/gemini-2.5-pro'."
            )
        for name, value in (
            ("infer.get_infer_batch_size()", self.get_infer_batch_size()),
            ("infer.num_drafts", self.infer["num_drafts"]),
            ("infer.concurrency", self.infer["concurrency"]),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1, but it is {value!r}.")

    def create_completion_settings(self) -> CompletionSettings:
        return CompletionSettings(
            self.infer["temperature"],
            self.infer["max_new_tokens"],
            self.infer["num_retries"],
            self.infer["request_timeout"],
        )

    def get_litellm_options(self) -> dict:
        return self.params.get("litellm", {})

    def get_max_context_tokens(self) -> int:
        return self.infer["max_context_tokens"]

    def get_concurrency(self) -> int:
        return self.infer["concurrency"]

    def drafts_would_be_identical(self, num_drafts: int) -> bool:
        return num_drafts > 1 and not self.infer["temperature"]

    def get_infer_batch_size(self) -> int:
        return self.infer["infer_batch_size"]

    def get_prompt(self) -> dict:
        return self.infer["prompt"]

    def _single_prompt_builder_for(self, has_corpus_block: bool) -> PromptBuilder[PromptMessages]:
        return self._corpus_prompt_builder if has_corpus_block else self.get_infer_prompt_builder()

    def _batch_prompt_builder_for(self, has_corpus_block: bool) -> BatchPromptBuilder:
        return self._corpus_batch_prompt_builder if has_corpus_block else self._batch_prompt_builder

    def build_messages(
        self,
        sources: Sequence[str],
        examples: Sequence[Example],
        src_lang: Language,
        trg_lang: Language,
        corpus_block: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        has_corpus_block = bool(corpus_block)
        selected = [] if has_corpus_block else examples
        if len(sources) == 1:
            builder = self._single_prompt_builder_for(has_corpus_block)
            messages = builder.build(sources[0], src_lang, trg_lang, examples=selected)
        else:
            messages = self._batch_prompt_builder_for(has_corpus_block).build_batch(
                sources, src_lang, trg_lang, examples=selected
            )
        if has_corpus_block:
            messages = messages.with_additional_context(corpus_block)
        return messages.to_chat_messages()

    def create_model(
        self,
        mixed_precision: bool = True,
        num_devices: int = 1,
        clearml_queue: Optional[str] = None,
        completion_client_factory: Optional[CompletionClientFactory] = None,
    ) -> NMTModel:
        if completion_client_factory is None:
            completion_client_factory = LiteLLMCompletionClientFactory()
        return RemoteLLMModel(self, completion_client_factory)


class RemoteLLMModel(NMTModel):
    # The train step writes this checkpoint so that CheckpointType.LAST resolves to step 1.
    _CHECKPOINT_STEP = 1
    _MODEL_INFO_FILENAME = "remote_llm_model.json"

    def __init__(
        self,
        config: RemoteLLMConfig,
        completion_client_factory: Optional[CompletionClientFactory] = None,
    ) -> None:
        super().__init__(config)
        self._config: RemoteLLMConfig = config
        self._client_factory = completion_client_factory or LiteLLMCompletionClientFactory()
        self._client: Optional[CompletionClient] = None
        self._corpus_block: Optional[str] = None
        # Requests run on a thread pool; guards the lazily built client and corpus block.
        self._lock = threading.Lock()

    def train(self) -> None:
        """Build the retrieval index; there is no fine-tuning. The index is only a cache, since
        experiment.py deletes the run directory unless --save-checkpoints is passed."""
        self._config.check_example_corpora()
        checkpoint_dir = self._checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        builder = self._config.get_infer_prompt_builder()
        info: Dict[str, Any] = {"model": self._config.model, "num_examples": builder.get_num_examples()}
        summary = self._config.summarize_example_pool()
        if summary is None:
            LOGGER.info("No in-context examples are configured, so there is no retrieval index to build.")
        elif builder.covers_whole_pool():
            rendered = builder.render_pool(
                self._config.language(self._config.get_train_src_iso()),
                self._config.language(self._config.get_train_trg_iso()),
            )
            corpus_tokens = self._get_client().count_tokens(rendered)
            info["num_training_pairs"] = summary.corpus_size
            info["corpus_tokens"] = corpus_tokens
            LOGGER.info(
                "num_examples covers the whole corpus: %d training examples (%s tokens) go in every request.",
                summary.corpus_size,
                corpus_tokens if corpus_tokens is not None else "an unknown number of",
            )
            self._warn_if_corpus_too_large(corpus_tokens)
        else:
            self._config.save_example_index(checkpoint_dir)
            info["num_training_pairs"] = summary.corpus_size
            info["retrieval_method"] = summary.selection_method
            LOGGER.info(
                "Built a %s retrieval index over %d training examples.",
                summary.selection_method,
                summary.corpus_size,
            )

        with (checkpoint_dir / self._MODEL_INFO_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)

    def save_effective_config(self, path: Path) -> None:
        # There are no training arguments to overlay, so the merged config is the effective one.
        with path.open("w") as file:
            yaml.dump(deepcopy(self._config.root), file)

    def _checkpoint_dir(self) -> Path:
        return self._config.model_dir / f"checkpoint-{self._CHECKPOINT_STEP}"

    def _warn_if_corpus_too_large(self, corpus_tokens: Optional[int]) -> None:
        limit = self._config.get_max_context_tokens()
        if corpus_tokens is not None and corpus_tokens > limit:
            LOGGER.warning(
                "The training corpus is %d tokens, which exceeds infer.max_context_tokens (%d). "
                "Requests may be rejected for exceeding the model's context window. Lower "
                "infer.prompt.num_examples so that only the most relevant examples are sent.",
                corpus_tokens,
                limit,
            )

    def _get_client(self) -> CompletionClient:
        with self._lock:
            if self._client is None:
                self._client = self._client_factory.create(self._config)
            return self._client

    def _get_corpus_block(self, src_lang: Language, trg_lang: Language) -> Optional[str]:
        """The whole corpus, for the system message, when num_examples covers all of it."""
        builder = self._config.get_infer_prompt_builder()
        if not builder.covers_whole_pool():
            return None
        # Taken before the lock, which _get_client() also acquires.
        client = self._get_client()
        with self._lock:
            if self._corpus_block is None:
                rendered = builder.render_pool(src_lang, trg_lang)
                self._warn_if_corpus_too_large(client.count_tokens(rendered))
                self._corpus_block = self._config.build_corpus_block(rendered)
            return self._corpus_block

    def _load_saved_index(self) -> None:
        self._config.check_example_corpora()
        self._config.load_example_index(self._checkpoint_dir())

    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> Generator[SentenceTranslationGroup, None, None]:
        self._load_saved_index()
        sentence_list = list(sentences)
        batches = self._batch_indices(len(sentence_list))
        yield from self._translate_batches(
            sentence_list,
            batches,
            self._config.language(src_iso),
            self._config.language(trg_iso),
            produce_multiple_translations,
        )

    def translate_test_files(
        self,
        input_paths: List[Path],
        translation_paths: List[Path],
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        ckpt: Union[CheckpointType, str, int] = CheckpointType.LAST,
    ) -> None:
        if save_confidences:
            # Fail before paying for inference, rather than on test.py's missing confidences file.
            self._check_confidences_supported()
        self._load_saved_index()

        default_src_iso = self._config.get_train_src_iso()
        default_trg_iso = self._config.get_train_trg_iso()
        for input_path, translation_path in zip(input_paths, translation_paths):
            src_iso, trg_iso = self._isos_for_test_file(input_path, default_src_iso, default_trg_iso)
            sentences = self._read_lines(input_path)
            batches = self._batch_indices(len(sentences))
            groups = list(
                self._translate_batches(
                    sentences,
                    batches,
                    self._config.language(src_iso),
                    self._config.language(trg_iso),
                    produce_multiple_translations,
                    save_confidences,
                )
            )
            draft_group = DraftGroup(groups)
            for draft_index, translated_draft in enumerate(draft_group.get_drafts(), 1):
                if produce_multiple_translations:
                    draft_path = translation_path.with_suffix(f".{draft_index}{translation_path.suffix}")
                else:
                    draft_path = translation_path
                with draft_path.open("w", encoding="utf-8", newline="\n") as out_file:
                    out_file.write("\n".join(translated_draft.get_all_tokenized_translations()) + "\n")
                if save_confidences:
                    generate_confidence_files(translated_draft, draft_path)

    def _check_confidences_supported(self) -> None:
        if self._config.get_infer_batch_size() != 1:
            raise RuntimeError(
                "Confidence scores are only available when each request translates a single "
                "segment, because a batched reply's token log probabilities cannot be attributed "
                "to individual segments. Set infer.get_infer_batch_size() to 1, or run without "
                "--save-confidences."
            )
        if not self._get_client().supports_logprobs():
            raise RuntimeError(
                f"Confidence scores are not available for '{self._config.model}', because the "
                "provider does not return token log probabilities (Anthropic models never do). "
                "Use a model that supports logprobs, such as an OpenAI, Azure, or Gemini "
                "model, or run without --save-confidences."
            )

    def _isos_for_test_file(self, input_path: Path, default_src_iso: str, default_trg_iso: str) -> Tuple[str, str]:
        match = re.match(r"^test\.([a-z]{2,3})\.([a-z]{2,3})\..*", input_path.name)
        if match:
            return match.group(1), match.group(2)
        return default_src_iso, default_trg_iso

    def _read_lines(self, path: Path) -> List[str]:
        with path.open("r", encoding="utf-8-sig") as file:
            return [line.strip() for line in file]

    def _batch_indices(self, count: int) -> List[List[int]]:
        size = max(1, self._config.get_infer_batch_size())
        return [list(range(start, min(start + size, count))) for start in range(0, count, size)]

    def _translate_batches(
        self,
        sentences: Sequence[str],
        batches: Sequence[Sequence[int]],
        src_lang: Language,
        trg_lang: Language,
        produce_multiple_translations: bool,
        want_logprobs: bool = False,
    ) -> Generator[SentenceTranslationGroup, None, None]:
        num_drafts = self.get_num_drafts() if produce_multiple_translations else 1
        if self._config.drafts_would_be_identical(num_drafts):
            LOGGER.warning(
                "infer.num_drafts is %d but infer.temperature is 0, so the drafts are likely to be "
                "identical. Raise the temperature to get varied drafts.",
                num_drafts,
            )

        # One slot per (draft, sentence). Tasks write to disjoint slots, so no lock is needed.
        results: List[List[Optional[Completion]]] = [[None] * len(sentences) for _ in range(num_drafts)]

        def run_task(task: Tuple[int, int]) -> None:
            batch_index, draft_index = task
            indices = batches[batch_index]
            completions = self._translate_batch(
                [sentences[i] for i in indices], src_lang, trg_lang, want_logprobs, usage
            )
            for index, completion in zip(indices, completions):
                results[draft_index][index] = completion

        usage = UsageTotals()
        tasks = [(batch_index, draft_index) for draft_index in range(num_drafts) for batch_index in range(len(batches))]
        concurrency = self._config.get_concurrency()
        if concurrency == 1 or len(tasks) <= 1:
            for task in tasks:
                run_task(task)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Consume the iterator so that any exception raised in a worker propagates here.
                list(executor.map(run_task, tasks))

        LOGGER.info("Translated %s segments using %s.", f"{len(sentences):,}", usage.describe())

        for index in range(len(sentences)):
            yield SentenceTranslationGroup(
                [(results[draft][index] or Completion("")).to_sentence_translation() for draft in range(num_drafts)]
            )

    def _translate_batch(
        self,
        texts: Sequence[str],
        src_lang: Language,
        trg_lang: Language,
        want_logprobs: bool = False,
        usage: Optional[UsageTotals] = None,
    ) -> List[Completion]:
        translations: List[Completion] = [Completion("")] * len(texts)
        # Blank segments are verses absent from the source; no request needed.
        non_blank = [(index, text) for index, text in enumerate(texts) if text.strip() != ""]
        if len(non_blank) == 0:
            return translations
        completed = self._complete_texts([text for _, text in non_blank], src_lang, trg_lang, want_logprobs, usage)
        for (index, _), completion in zip(non_blank, completed):
            translations[index] = completion
        return translations

    def _complete_texts(
        self,
        texts: Sequence[str],
        src_lang: Language,
        trg_lang: Language,
        want_logprobs: bool = False,
        usage: Optional[UsageTotals] = None,
    ) -> List[Completion]:
        """Recovers from a miscounted reply by correcting, then halving the batch, then falling
        back to one request per segment. Only that last case can carry log probabilities, since a
        batched reply's token stream cannot be split per segment."""
        if len(texts) == 1:
            return [self._complete_single(texts[0], src_lang, trg_lang, want_logprobs, usage)]

        messages = self._build_messages(texts, src_lang, trg_lang)
        response = self._complete(messages, usage=usage).text
        parsed = ModelReply(response).parse(len(texts))
        if parsed is None:
            correction = messages + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": (
                        f"That reply did not have the required format. Reply again with exactly "
                        f"{len(texts)} lines, one per segment, in the same order, each formatted as "
                        f"`<number>. <translation>`, and nothing else."
                    ),
                },
            ]
            parsed = ModelReply(self._complete(correction, usage=usage).text).parse(len(texts))
        if parsed is not None:
            return [Completion(text) for text in parsed]

        LOGGER.warning(
            "Could not read %d translations from the model's reply; splitting the batch and retrying.", len(texts)
        )
        middle = len(texts) // 2
        return self._complete_texts(texts[:middle], src_lang, trg_lang, want_logprobs, usage) + self._complete_texts(
            texts[middle:], src_lang, trg_lang, want_logprobs, usage
        )

    def _complete_single(
        self,
        text: str,
        src_lang: Language,
        trg_lang: Language,
        want_logprobs: bool = False,
        usage: Optional[UsageTotals] = None,
    ) -> Completion:
        completion = self._complete(self._build_messages([text], src_lang, trg_lang), want_logprobs, usage)
        stripped = ModelReply(completion.text).strip_code_fence().strip()
        if stripped == completion.text:
            return completion
        # The scores still cover the stripped text, so drop them rather than misalign them.
        return replace(completion, text=stripped, token_logprobs=[])

    def _complete(
        self, messages: List[Dict[str, str]], logprobs: bool = False, usage: Optional[UsageTotals] = None
    ) -> Completion:
        """Send one request, counting it against the run's totals."""
        completion = self._get_client().complete(messages, logprobs)
        if usage is not None:
            usage.add(completion)
        return completion

    def _build_messages(self, texts: Sequence[str], src_lang: Language, trg_lang: Language) -> List[Dict[str, str]]:
        corpus_block = self._get_corpus_block(src_lang, trg_lang)
        examples = [] if corpus_block else self._retrieve_examples(texts)
        return self._config.build_messages(texts, examples, src_lang, trg_lang, corpus_block)

    def _retrieve_examples(self, texts: Sequence[str]) -> List[Example]:
        return self._config.get_infer_prompt_builder().select_examples("\n".join(texts))
