"""In-context learning translation with a hosted LLM, via LiteLLM.

An implementation of the :class:`Config`/:class:`NMTModel` abstractions that does **no
fine-tuning**: it translates by prompting a hosted LLM with parallel examples drawn from the
training corpus. Calls go through LiteLLM, so one ``model`` string selects any supported
provider (Anthropic, OpenAI, Gemini, Bedrock, a local server, ...).

Setting ``data.tokenize: false`` keeps the model-agnostic parts of the pipeline usable as they
are: preprocessing writes raw detokenized parallel text, and evaluation
(:mod:`silnlp.nmt.test`) and inference orchestration (:mod:`silnlp.nmt.translate`) work
unchanged.

Three things are configurable:

* **Examples** (``infer.prompt.num_examples``): how many parallel examples go in each prompt. A
  number at or above the corpus size puts the whole corpus in every prompt.
* **Selection method** (``infer.prompt.example_selection.method``): ``tfidf``, ``bm25`` or
  ``embedding``; see :mod:`silnlp.nmt.example_retrieval`.
* **Batch size** (``infer.infer_batch_size``): how many consecutive segments go in one request.

The default prompts are written for scripture: they cast the model as a member of the
translation team and have it infer the team's style, key terms, exegesis, and orthography from
the examples, which are that team's own work. All are overridable through ``infer.prompt``.

The train step does no fine-tuning, but it is not a no-op: it builds the retrieval index and
writes ``run/checkpoint-1``, so the checkpoint machinery the test and translate steps rely on
resolves normally.

Confidence scores come from the provider's token log probabilities. They need a provider that
returns them (OpenAI, Azure, and Gemini do; Anthropic never does) and one segment per request; see
``RemoteLLMModel._check_confidences_supported``.
"""

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
from .example_retrieval import TFIDF_METHOD, Example
from .llm_config import LLMConfig, PromptBuilder

LOGGER = logging.getLogger(__name__)

# The train step writes this checkpoint so that CheckpointType.LAST resolves to step 1.
CHECKPOINT_STEP = 1
MODEL_INFO_FILENAME = "remote_llm_model.json"

# The prompts cast the model as a member of the translation team, because consistency with this
# team's decisions matters more than general translation competence. All are overridable through
# infer.prompt.
DEFAULT_SYSTEM_MESSAGE_TEMPLATE = (
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
SINGLE_INSTRUCTION = (
    "Translate this {src_lang} passage into {trg_lang} as the team would translate it. Reply with "
    "only the translation.\n\n{source}"
)
BATCH_INSTRUCTION = (
    "Translate the following {num_segments} consecutive {src_lang} passages into {trg_lang} as the "
    "team would translate them. Some may be section headings rather than verses. Read them "
    "together, so that participants, pronouns, and the flow of the passage stay consistent across "
    "them, but translate each one on its own.\n"
    "Reply with exactly {num_segments} lines, one per passage, in the same order, each formatted "
    "as `<number>. <translation>`. Do not merge, split, reorder, or omit passages, and do not add "
    "any other text.\n\n{source}"
)
EXAMPLES_HEADING = (
    "The team has already translated these passages. They are your model for this team's style, "
    "terminology, and exegesis:\n\n{examples}"
)
CORPUS_HEADING = (
    "This is everything the team has translated so far. It is your reference for this team's "
    "style, terminology, exegesis, spelling, and punctuation:"
)

DEFAULT_SINGLE_INSTRUCTION_TEMPLATE = SINGLE_INSTRUCTION
DEFAULT_BATCH_INSTRUCTION_TEMPLATE = BATCH_INSTRUCTION
DEFAULT_FEW_SHOT_SINGLE_INSTRUCTION_TEMPLATE = EXAMPLES_HEADING + SINGLE_INSTRUCTION
DEFAULT_FEW_SHOT_BATCH_INSTRUCTION_TEMPLATE = EXAMPLES_HEADING + BATCH_INSTRUCTION
DEFAULT_EXAMPLE_FORMAT = {"type": "text", "template": "{src_lang}: {source}\n{trg_lang}: {target}\n\n"}

_CODE_FENCE = re.compile(r"^\s*```[^\n]*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_NUMBERED_LINE = re.compile(r"^\s*(\d{1,4})\s*[.):\]]\s*(.*)$")


def strip_code_fence(text: str) -> str:
    match = _CODE_FENCE.match(text.strip())
    return match.group(1) if match is not None else text


def parse_numbered_response(text: str, num_segments: int) -> Optional[List[str]]:
    """Parse a numbered list of translations, or return None if the reply is malformed.

    A None return is the signal for the caller's recovery ladder (correct, then split the batch,
    then fall back to one request per segment). Lines that are not numbered are treated as
    continuations of the preceding translation, and any preamble before the first numbered line
    is ignored.
    """
    if num_segments <= 0:
        return []
    parsed: Dict[int, List[str]] = {}
    current: Optional[int] = None
    for line in strip_code_fence(text).splitlines():
        match = _NUMBERED_LINE.match(line)
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


def group_indices_by_size(count: int, size: int) -> List[List[int]]:
    size = max(1, size)
    return [list(range(start, min(start + size, count))) for start in range(0, count, size)]


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float


@dataclass(frozen=True)
class Completion:
    """A reply from the model, with whatever the provider reported alongside it.

    ``cost`` is None when LiteLLM has no pricing for the model, which is not the same as free.
    """

    text: str
    token_logprobs: List[TokenLogprob] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: Optional[float] = None

    def mean_logprob(self) -> Optional[float]:
        if len(self.token_logprobs) == 0:
            return None
        return sum(entry.logprob for entry in self.token_logprobs) / len(self.token_logprobs)


@dataclass
class UsageTotals:
    """Running totals for one translation run. Requests are made from several threads."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    unpriced_requests: int = 0

    def __post_init__(self) -> None:
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


class CompletionClient(ABC):
    """Sends a chat completion request and returns the reply text.

    Indirected so tests can substitute a scripted client without a network call.
    """

    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        ...

    def supports_logprobs(self) -> bool:
        """Whether the configured provider can return token log probabilities."""
        return False


class LiteLLMCompletionClient(CompletionClient):
    def __init__(self, model: str, infer: dict, extra_kwargs: Optional[dict] = None) -> None:
        self._model = model
        self._infer = infer
        self._extra_kwargs: Dict[str, Any] = dict(extra_kwargs or {})

    def complete(self, messages: List[Dict[str, str]], logprobs: bool = False) -> Completion:
        litellm = _import_litellm()
        extra_kwargs = dict(self._extra_kwargs)
        if logprobs:
            extra_kwargs["logprobs"] = True
        response = litellm.completion(
            model=self._model,
            messages=messages,
            temperature=self._infer["temperature"],
            max_tokens=self._infer["max_new_tokens"],
            # LiteLLM retries transient failures (rate limits, timeouts, 5xx) itself.
            num_retries=self._infer["num_retries"],
            timeout=self._infer["request_timeout"],
            **extra_kwargs,
        )
        choice = response["choices"][0]
        content = choice["message"]["content"]
        usage = _get_field(response, "usage")
        return Completion(
            content if content is not None else "",
            extract_token_logprobs(choice) if logprobs else [],
            prompt_tokens=int(_get_field(usage, "prompt_tokens") or 0),
            completion_tokens=int(_get_field(usage, "completion_tokens") or 0),
            cost=_completion_cost(litellm, response),
        )

    def supports_logprobs(self) -> bool:
        litellm = _import_litellm()
        try:
            supported = litellm.get_supported_openai_params(self._model) or []
        except Exception:
            # An unrecognized model is not fatal; the caller just gets no scores.
            LOGGER.warning("Could not determine which parameters %s supports.", self._model, exc_info=True)
            return False
        return "logprobs" in supported


def extract_token_logprobs(choice: Any) -> List[TokenLogprob]:
    """Pull the per-token log probabilities out of a LiteLLM choice, if it has any.

    A provider that does not support logprobs omits them rather than failing, so every level of
    the normalized OpenAI shape has to be treated as optional.
    """
    logprobs = _get_field(choice, "logprobs")
    content = _get_field(logprobs, "content") if logprobs is not None else None
    if not content:
        return []
    token_logprobs: List[TokenLogprob] = []
    for entry in content:
        token = _get_field(entry, "token")
        logprob = _get_field(entry, "logprob")
        if token is None or logprob is None:
            continue
        token_logprobs.append(TokenLogprob(str(token), float(logprob)))
    return token_logprobs


def _completion_cost(litellm: Any, response: Any) -> Optional[float]:
    """What the request cost, or None if LiteLLM has no pricing for the model."""
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception:
        LOGGER.debug("No pricing available for this response; reporting its cost as unknown.", exc_info=True)
        return None


def _get_field(obj: Any, name: str) -> Any:
    """Read a field from a LiteLLM response, which may be a dict or a pydantic model."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def count_tokens(model: str, text: str) -> Optional[int]:
    """Count the tokens in ``text`` the way ``model``'s tokenizer would, or None if it cannot.

    None means the caller skips its size check rather than run it against a guess. LiteLLM
    falls back to a default tokenizer for models it does not recognize, so this rarely happens.
    """
    try:
        import litellm

        return int(litellm.token_counter(model=model, text=text))
    except Exception:
        LOGGER.warning("Could not count tokens for '%s'; skipping the prompt size check.", model, exc_info=True)
        return None


def _import_litellm():
    try:
        import litellm
    except ImportError as e:
        raise ImportError(
            "Remote LLM experiments require the 'litellm' package, which is part of the "
            "'remote_llm' extra. Install it with `poetry install -E remote_llm`."
        ) from e
    return litellm


class CompletionClientFactory:
    def create(self, config: "RemoteLLMConfig") -> CompletionClient:
        raise NotImplementedError


class LiteLLMCompletionClientFactory(CompletionClientFactory):
    def create(self, config: "RemoteLLMConfig") -> CompletionClient:
        return LiteLLMCompletionClient(config.model, config.infer, config.params.get("litellm"))


class RemoteLLMConfig(LLMConfig):
    REQUIRE_EXAMPLE_CORPUS = False
    DEFAULT_SYSTEM_MESSAGE = DEFAULT_SYSTEM_MESSAGE_TEMPLATE
    DEFAULT_INSTRUCTION_TEMPLATE = DEFAULT_SINGLE_INSTRUCTION_TEMPLATE
    DEFAULT_FEW_SHOT_INSTRUCTION_TEMPLATE = DEFAULT_FEW_SHOT_SINGLE_INSTRUCTION_TEMPLATE
    DEFAULT_EXAMPLE_FORMAT = DEFAULT_EXAMPLE_FORMAT

    def __init__(self, exp_dir: Path, config: dict, environment: SilNlpEnv) -> None:
        super().__init__(exp_dir, config, environment)
        self._validate()
        self._batch_prompt_builder = self._create_batch_prompt_builder()
        self._disable_eval_if_no_val_split()

    def _default_config(self, exp_dir: Path) -> dict:
        return merge_dict(
            super()._default_config(exp_dir),
            {
                "train": {
                    "output_dir": str(exp_dir / "run"),
                },
                # No training loop, so none of this applies; the keys exist because the shared
                # config plumbing reads and writes them.
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
                        # TF-IDF by default so a plain install works; BM25 generally ranks
                        # better but needs rank_bm25 from the 'llm' extra.
                        "example_selection": {"method": TFIDF_METHOD, "model": None},
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

    def _example_corpus_paths(self) -> List[Tuple[Path, Path]]:
        """Prefer the detokenized corpus, so an experiment preprocessed for a tokenized model
        still yields readable examples."""
        detokenized = (
            self.exp_dir / self.train_src_detok_filename(),
            self.exp_dir / self.train_trg_detok_filename(),
        )
        return [detokenized, *super()._example_corpus_paths()]

    def _resolve_infer_prompt_defaults(self, prompt: dict) -> None:
        # A hoisted corpus brings its own heading, so keep the wording that has none for it to
        # use; the few-shot defaults would otherwise head an examples block that renders empty.
        self._corpus_single_instruction_template = (
            prompt.get("instruction_template") or DEFAULT_SINGLE_INSTRUCTION_TEMPLATE
        )
        self._corpus_batch_instruction_template = (
            prompt.get("batch_instruction_template") or DEFAULT_BATCH_INSTRUCTION_TEMPLATE
        )
        super()._resolve_infer_prompt_defaults(prompt)
        if prompt["batch_instruction_template"] is None:
            prompt["batch_instruction_template"] = (
                DEFAULT_FEW_SHOT_BATCH_INSTRUCTION_TEMPLATE
                if int(prompt["num_examples"]) > 0
                else DEFAULT_BATCH_INSTRUCTION_TEMPLATE
            )

    def _create_batch_prompt_builder(self) -> PromptBuilder:
        """A second builder whose instruction template numbers several segments in one request."""
        batch_prompt = dict(self.prompt)
        batch_prompt["instruction_template"] = self.prompt["batch_instruction_template"]
        return self._create_prompt_builder(batch_prompt, "infer.prompt.batch_instruction_template")

    def _validate(self) -> None:
        if not str(self.model).strip():
            raise ValueError(
                "An in-context learning experiment needs a 'model' in LiteLLM format, "
                "e.g. 'anthropic/claude-sonnet-4-5', 'gpt-4o', or 'gemini/gemini-2.5-pro'."
            )
        for name, value in (
            ("infer.infer_batch_size", self.infer_batch_size),
            ("infer.num_drafts", self.infer["num_drafts"]),
            ("infer.concurrency", self.infer["concurrency"]),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1, but it is {value!r}.")

    @property
    def num_examples(self) -> int:
        return self.infer_prompt_builder.num_examples

    @property
    def infer_batch_size(self) -> int:
        return self.infer["infer_batch_size"]

    @property
    def prompt(self) -> dict:
        return self.infer["prompt"]

    def prompt_builder_for(self, num_segments: int) -> PromptBuilder:
        return self.infer_prompt_builder if num_segments == 1 else self._batch_prompt_builder

    def build_messages(
        self,
        sources: Sequence[str],
        examples: Sequence[Example],
        src_lang: Language,
        trg_lang: Language,
        corpus_block: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the chat messages for one translation request.

        When the whole corpus is in play it goes in the system message, ahead of everything that
        varies per request, so the prompt prefix is byte-identical across requests and
        provider-side prompt caching can apply.
        """
        builder = self.prompt_builder_for(len(sources))
        if len(sources) == 1:
            source, extra_fields = sources[0], {}
        else:
            source = "\n".join(f"{i}. {text}" for i, text in enumerate(sources, 1))
            extra_fields = {"num_segments": len(sources)}
        template = builder.template_for(None)
        if corpus_block:
            instruction_template = (
                self._corpus_single_instruction_template
                if len(sources) == 1
                else self._corpus_batch_instruction_template
            )
        else:
            instruction_template = template.instruction_template
        instruction = instruction_template.format(
            src_lang=src_lang.name,
            trg_lang=trg_lang.name,
            source=source,
            examples="" if corpus_block else builder.render_examples(examples, src_lang, trg_lang),
            **extra_fields,
        )
        system_message = template.system_message.format(src_lang=src_lang.name, trg_lang=trg_lang.name)
        if corpus_block:
            system_message = f"{system_message}\n\n{corpus_block}" if system_message else corpus_block
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": instruction})
        return messages

    def render_examples(self, examples: Sequence[Example], src_lang: Language, trg_lang: Language) -> str:
        return self.infer_prompt_builder.render_examples(examples, src_lang, trg_lang)

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
        # Requests run on a thread pool; guards the lazily built client, index, and corpus block.
        self._lock = threading.Lock()

    def train(self) -> None:
        """Build the retrieval index. There is no fine-tuning.

        The saved index is only a cache: ``experiment.py`` deletes the ``run`` directory unless
        ``--save-checkpoints`` is passed, so inference rebuilds it when it is missing.
        """
        checkpoint_dir = self._checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        builder = self._config.infer_prompt_builder
        info: Dict[str, Any] = {"model": self._config.model, "num_examples": builder.num_examples}
        pool = builder.pool
        if pool is None:
            LOGGER.info("No in-context examples are configured, so there is no retrieval index to build.")
        elif builder.covers_whole_pool():
            rendered = self._config.render_examples(
                pool.examples,
                self._config.language(self._config.train_src_iso),
                self._config.language(self._config.train_trg_iso),
            )
            corpus_tokens = count_tokens(self._config.model, rendered)
            info["num_training_pairs"] = len(pool)
            info["corpus_tokens"] = corpus_tokens
            LOGGER.info(
                "num_examples covers the whole corpus: %d training examples (%s tokens) go in every request.",
                len(pool),
                corpus_tokens if corpus_tokens is not None else "an unknown number of",
            )
            self._warn_if_corpus_too_large(corpus_tokens)
        else:
            pool.save_index(checkpoint_dir)
            info["num_training_pairs"] = len(pool)
            info["retrieval_method"] = pool.method
            LOGGER.info("Built a %s retrieval index over %d training examples.", pool.method, len(pool))

        with (checkpoint_dir / MODEL_INFO_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)

    def save_effective_config(self, path: Path) -> None:
        # There are no training arguments to overlay, so the merged config is the effective one.
        with path.open("w") as file:
            yaml.dump(deepcopy(self._config.root), file)

    def _checkpoint_dir(self) -> Path:
        return self._config.model_dir / f"checkpoint-{CHECKPOINT_STEP}"

    def _warn_if_corpus_too_large(self, corpus_tokens: Optional[int]) -> None:
        limit: int = self._config.infer["max_context_tokens"]
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
        builder = self._config.infer_prompt_builder
        if builder.pool is None or not builder.covers_whole_pool():
            return None
        with self._lock:
            if self._corpus_block is None:
                rendered = self._config.render_examples(builder.pool.examples, src_lang, trg_lang)
                self._warn_if_corpus_too_large(count_tokens(self._config.model, rendered))
                self._corpus_block = f"{CORPUS_HEADING}\n\n{rendered}" if rendered else ""
            return self._corpus_block

    def _load_saved_index(self) -> None:
        pool = self._config.infer_prompt_builder.pool
        if pool is not None:
            pool.load_index(self._checkpoint_dir())

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
            # Fail before paying for inference; test.py would otherwise fail later with a
            # confusing FileNotFoundError for the missing confidences file.
            self._check_confidences_supported()
        self._load_saved_index()

        default_src_iso = self._config.train_src_iso
        default_trg_iso = self._config.train_trg_iso
        for input_path, translation_path in zip(input_paths, translation_paths):
            src_iso, trg_iso = self._isos_for_test_file(input_path, default_src_iso, default_trg_iso)
            sentences = _read_lines(input_path)
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
        """Raise unless the provider returns log probabilities and each request is one segment.

        LiteLLM omits logprobs silently rather than failing, so an unsupported provider has to be
        caught here.
        """
        if self._config.infer_batch_size != 1:
            raise RuntimeError(
                "Confidence scores are only available when each request translates a single "
                "segment, because a batched reply's token log probabilities cannot be attributed "
                "to individual segments. Set infer.infer_batch_size to 1, or run without "
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

    def _batch_indices(self, count: int) -> List[List[int]]:
        return group_indices_by_size(count, self._config.infer_batch_size)

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
        if num_drafts > 1 and not self._config.infer["temperature"]:
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
        concurrency = max(1, self._config.infer["concurrency"])
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
                [_to_sentence_translation(results[draft][index] or Completion("")) for draft in range(num_drafts)]
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
        """Translate a batch, recovering from a reply that does not have one line per segment.

        The ladder is: ask again with a correction, then split the batch in half, and finally
        fall back to one request per segment, which cannot be miscounted. Log probabilities are
        only attached in that last case, since a batched reply's token stream cannot be split
        per segment.
        """
        if len(texts) == 1:
            return [self._complete_single(texts[0], src_lang, trg_lang, want_logprobs, usage)]

        messages = self._build_messages(texts, src_lang, trg_lang)
        response = self._complete(messages, usage=usage).text
        parsed = parse_numbered_response(response, len(texts))
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
            parsed = parse_numbered_response(self._complete(correction, usage=usage).text, len(texts))
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
        stripped = strip_code_fence(completion.text).strip()
        if stripped == completion.text:
            return completion
        # Stripping a code fence or surrounding whitespace leaves the per-token scores covering
        # text that is no longer there, so drop them rather than report a misaligned sequence.
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
        return self._config.infer_prompt_builder.select_examples("\n".join(texts))


def _to_sentence_translation(completion: Completion) -> SentenceTranslation:
    """Convert a completion into the pipeline's translation type.

    ``tokens`` holds the whole translation rather than the provider's subword tokens: the
    predictions file is raw text and is written by space-joining them. The log probabilities go
    into the scores, which is all the confidence files read.
    """
    return SentenceTranslation(
        completion.text,
        [completion.text],
        [entry.logprob for entry in completion.token_logprobs],
        completion.mean_logprob(),
        starts_with_special_token=False,
    )


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig") as file:
        return [line.strip() for line in file]
