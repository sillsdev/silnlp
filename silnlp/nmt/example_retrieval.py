"""Few-shot examples for LLM translation prompts: retrieval (ExampleRetriever), formatting
(ExampleFormatter), and prompt assembly (PromptExampleConfig, ExamplePromptBuilder)."""

import json
import logging
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Union
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from .corpora import read_parallel_text_pairs

LOGGER = logging.getLogger(__name__)

TFIDF_METHOD = "tfidf"
BM25_METHOD = "bm25"
EMBEDDING_METHOD = "embedding"
VALID_SELECTION_METHODS = (TFIDF_METHOD, BM25_METHOD, EMBEDDING_METHOD)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RETRIEVER_FILENAME = "retrieval.pkl"
RETRIEVER_META_FILENAME = "retrieval_meta.json"

# Shared by tfidf and bm25 so that switching between them doesn't also change tokenization.
_TOKEN_PATTERN = r"\w+"

# TranslateGemma's chat template requires structured {type, lang_code, text} content instead of
# free text, so it cannot carry few-shot examples.
TRANSLATE_GEMMA_MODEL_PREFIXES = ("google/translate-gemma", "google/translategemma")


def tokenize_for_retrieval(text: str) -> List[str]:
    return re.findall(_TOKEN_PATTERN, text.lower())


@dataclass(frozen=True)
class Example:
    source: str
    target: str


class _EmbeddingModel(Protocol):
    """SentenceTransformer's interface, narrowed to what's used here; also the test injection point."""

    def encode(
        self, texts: Sequence[str], convert_to_numpy: bool, normalize_embeddings: bool, show_progress_bar: bool
    ) -> np.ndarray: ...


def _top_k_indices(scores: np.ndarray, k: int, exclude: Optional[int] = None) -> List[int]:
    """Indices of the top-k highest scores, most-similar first, optionally excluding one index."""
    n = scores.shape[0]
    # Reduces k, not just the excluded index's score, so the excluded index (-inf below) can
    # never be picked even when k would otherwise cover the whole pool.
    available = n - 1 if exclude is not None else n
    k = min(k, available)
    if k <= 0:
        return []
    if exclude is not None:
        scores = scores.copy()
        scores[exclude] = -np.inf
    top = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    return top[np.argsort(-scores[top])].tolist()


class ExampleRetriever(ABC):
    """A fitted index over the source side of an example pool."""

    method: str = ""

    def __init__(self) -> None:
        self._examples: List[Example] = []

    def __len__(self) -> int:
        return len(self._examples)

    @property
    def examples(self) -> List[Example]:
        return self._examples

    @property
    def model_name(self) -> Optional[str]:
        return None

    def fit(self, examples: Sequence[Example]) -> None:
        self._examples = list(examples)
        self._fit_index([example.source for example in self._examples])

    def retrieve(self, query: str, k: int) -> List[Example]:
        """Top-k most similar pool examples to an arbitrary query string not in the pool
        (used at eval/test/translate time)."""
        if k <= 0 or len(self._examples) == 0:
            return []
        return [self._examples[i] for i in self._top_indices_for_query(query, k)]

    def retrieve_for_pool_index(self, index: int, k: int) -> List[Example]:
        """Top-k most similar pool examples to the pool entry at `index`, excluding itself
        (leave-one-out; used during training, where the pool contains the row being translated)."""
        if k <= 0 or len(self._examples) == 0:
            return []
        return [self._examples[i] for i in self._top_indices_for_pool_index(index, k)]

    @abstractmethod
    def _fit_index(self, sources: List[str]) -> None: ...

    @abstractmethod
    def _top_indices_for_query(self, query: str, k: int) -> List[int]: ...

    def _top_indices_for_pool_index(self, index: int, k: int) -> List[int]:
        # Asks for k + 1 so that dropping the pool entry itself still leaves k results.
        indices = self._top_indices_for_query(self._examples[index].source, k + 1)
        return [i for i in indices if i != index][:k]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / RETRIEVER_FILENAME).open("wb") as file:
            pickle.dump(self, file)
        meta = {"method": self.method, "model_name": self.model_name, "num_examples": len(self._examples)}
        with (directory / RETRIEVER_META_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)

    @staticmethod
    def load(directory: Path) -> Optional["ExampleRetriever"]:
        """Load a previously saved index, or return None if it is missing or unreadable.

        A None return is not an error: the caller rebuilds. The index often will not be there,
        since the ``run`` directory is deleted unless ``--save-checkpoints`` is set, and it may
        not unpickle across a scikit-learn upgrade.
        """
        path = directory / RETRIEVER_FILENAME
        try:
            with path.open("rb") as file:
                retriever = pickle.load(file)
        except FileNotFoundError:
            return None
        except Exception:
            LOGGER.warning("Could not load the retrieval index at %s; it will be rebuilt.", path, exc_info=True)
            return None
        if not isinstance(retriever, ExampleRetriever):
            LOGGER.warning("The file at %s is not a retrieval index; it will be rebuilt.", path)
            return None
        return retriever


class TfidfExampleRetriever(ExampleRetriever):
    method = TFIDF_METHOD

    def __init__(self) -> None:
        super().__init__()
        self._vectorizer: Optional[Any] = None
        self._matrix: Optional[Any] = None

    def _fit_index(self, sources: List[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if len(sources) == 0:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(lowercase=True, token_pattern=_TOKEN_PATTERN)
        # Rows are L2-normalized by default, so a dot product against the matrix is cosine similarity.
        self._matrix = self._vectorizer.fit_transform(sources)

    def _scores_for_vector(self, vector) -> np.ndarray:
        return (self._matrix @ vector.T).toarray().ravel()

    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        if self._vectorizer is None or self._matrix is None:
            return []
        return _top_k_indices(self._scores_for_vector(self._vectorizer.transform([query])), k)

    def _top_indices_for_pool_index(self, index: int, k: int) -> List[int]:
        # Guarded by retrieve_for_pool_index()'s empty-pool check, so this is always fit here.
        assert self._matrix is not None
        return _top_k_indices(self._scores_for_vector(self._matrix[index]), k, exclude=index)


class BM25ExampleRetriever(ExampleRetriever):
    method = BM25_METHOD

    def __init__(self) -> None:
        super().__init__()
        self._index: Optional[Any] = None

    def _fit_index(self, sources: List[str]) -> None:
        bm25_okapi = _import_bm25()
        tokenized = [tokenize_for_retrieval(source) for source in sources]
        # BM25Okapi rejects an empty corpus and divides by zero on all-empty documents.
        if sum(len(tokens) for tokens in tokenized) == 0:
            self._index = None
            return
        self._index = bm25_okapi(tokenized)

    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        if self._index is None:
            return []
        return _top_k_indices(self._index.get_scores(tokenize_for_retrieval(query)), k)


class EmbeddingExampleRetriever(ExampleRetriever):
    method = EMBEDDING_METHOD

    def __init__(self, model_name: Optional[str] = None, model: Optional[_EmbeddingModel] = None) -> None:
        """`model` is the test injection seam; production passes only `model_name`."""
        super().__init__()
        self._model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self._model = model
        self._embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def _get_model(self) -> _EmbeddingModel:
        if self._model is None:
            self._model = _load_sentence_transformer(self._model_name)
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._get_model().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    def _fit_index(self, sources: List[str]) -> None:
        self._embeddings = self._encode(sources) if sources else np.zeros((0, 0), dtype=np.float32)

    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        if self._embeddings.shape[0] == 0:
            return []
        return _top_k_indices(self._embeddings @ self._encode([query])[0], k)

    def _top_indices_for_pool_index(self, index: int, k: int) -> List[int]:
        return _top_k_indices(self._embeddings @ self._embeddings[index], k, exclude=index)

    def __getstate__(self) -> dict:
        # The embeddings are the expensive part worth caching; the model reloads by name.
        return {**self.__dict__, "_model": None}


def _import_bm25():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError(
            f"example_selection.method: {BM25_METHOD} requires the 'rank_bm25' package. "
            "Install it with `poetry install -E llm`."
        ) from e
    return BM25Okapi


def _load_sentence_transformer(model_name: str) -> _EmbeddingModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            f"example_selection.method: {EMBEDDING_METHOD} requires the 'sentence-transformers' "
            "package. Install it with `poetry install -E llm`."
        ) from e
    return SentenceTransformer(model_name)


def create_example_retriever(method: str, model_name: Optional[str] = None) -> ExampleRetriever:
    """Create an unfitted retriever; the caller supplies the pool with `fit`."""
    normalized = method.lower()
    if normalized == TFIDF_METHOD:
        return TfidfExampleRetriever()
    if normalized == BM25_METHOD:
        return BM25ExampleRetriever()
    if normalized == EMBEDDING_METHOD:
        return EmbeddingExampleRetriever(model_name)
    raise ValueError(
        f"Unknown example_selection.method '{method}'. Valid options: {', '.join(VALID_SELECTION_METHODS)}."
    )


class ExampleFormatter(ABC):
    """Renders retrieved examples into the text that fills {examples} in instruction_template."""

    @abstractmethod
    def format(self, examples: Sequence[Example], src_lang_name: str, trg_lang_name: str) -> str: ...


class TextExampleFormatter(ExampleFormatter):
    """Unlike JsonExampleFormatter/XmlExampleFormatter, does not escape the template output."""

    DEFAULT_TEMPLATE = "Source ({src_lang}): {source}\nTranslation ({trg_lang}): {target}\n\n"

    def __init__(self, template: str = DEFAULT_TEMPLATE) -> None:
        self._template = template

    def format(self, examples: Sequence[Example], src_lang_name: str, trg_lang_name: str) -> str:
        return "".join(
            self._template.format(src_lang=src_lang_name, trg_lang=trg_lang_name, source=ex.source, target=ex.target)
            for ex in examples
        )


class JsonExampleFormatter(ExampleFormatter):
    def format(self, examples: Sequence[Example], src_lang_name: str, trg_lang_name: str) -> str:
        if not examples:
            return ""
        # Fixed field names (not language-named) so the schema is stable across experiments.
        payload = [{"source": ex.source, "target": ex.target} for ex in examples]
        # ensure_ascii=False avoids \uXXXX-escaping non-Latin target text.
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


class XmlExampleFormatter(ExampleFormatter):
    def format(self, examples: Sequence[Example], src_lang_name: str, trg_lang_name: str) -> str:
        return "".join(
            f"<example>\n<source>{xml_escape(ex.source)}</source>\n<target>{xml_escape(ex.target)}</target>\n"
            "</example>\n"
            for ex in examples
        )


def create_example_formatter(format_params: Union[str, dict]) -> ExampleFormatter:
    # A bare string is shorthand for {"type": <string>}
    if isinstance(format_params, str):
        format_params = {"type": format_params}
    format_type = str(format_params.get("type", "text")).lower()
    if format_type == "text":
        return TextExampleFormatter(format_params.get("template", TextExampleFormatter.DEFAULT_TEMPLATE))
    if format_type == "json":
        return JsonExampleFormatter()
    if format_type == "xml":
        return XmlExampleFormatter()
    raise ValueError(f"Unknown example_format.type '{format_type}'. Valid options: text, json, xml.")


class PromptExampleConfig:
    """Parsed prompt config for few-shot examples."""

    def __init__(
        self,
        num_examples: int,
        formatter: ExampleFormatter,
        selection_method: str,
        selection_model: Optional[str],
        instruction_template: str,
        model: str,
    ) -> None:
        if num_examples < 0:
            raise ValueError(f"prompt.num_examples must be non-negative, got {num_examples}.")

        selection_method = selection_method.lower()
        if selection_method not in VALID_SELECTION_METHODS:
            raise ValueError(
                f"Unknown example_selection.method '{selection_method}'. "
                f"Valid options: {', '.join(VALID_SELECTION_METHODS)}."
            )

        if num_examples > 0:
            if "{examples}" not in instruction_template:
                LOGGER.warning(
                    "prompt.num_examples > 0 requires '{examples}' in prompt.instruction_template, "
                    "otherwise the retrieved examples are silently discarded."
                )
            if model.lower().startswith(TRANSLATE_GEMMA_MODEL_PREFIXES):
                raise RuntimeError(
                    "TranslateGemma models do not support few-shot examples in the prompt. "
                    "Set prompt.num_examples to 0 or use a different model."
                )

        self.num_examples = num_examples
        self.formatter = formatter
        self.selection_method = selection_method
        self.selection_model = selection_model

    @staticmethod
    def from_params(prompt_params: dict, model: str) -> "PromptExampleConfig":
        example_selection = prompt_params["example_selection"]
        if isinstance(example_selection, str):
            example_selection = {"method": example_selection}
        return PromptExampleConfig(
            num_examples=int(prompt_params["num_examples"]),
            formatter=create_example_formatter(prompt_params["example_format"]),
            selection_method=str(example_selection["method"]),
            selection_model=example_selection.get("model"),
            instruction_template=prompt_params["instruction_template"],
            model=model,
        )


class ExamplePromptBuilder:
    """Builds the {examples} text block for a translation prompt."""

    def __init__(self, config: PromptExampleConfig, pool_src_path: Path, pool_trg_path: Path) -> None:
        self.config = config
        self._pool_src_path = pool_src_path
        self._pool_trg_path = pool_trg_path
        self._retriever: Optional[ExampleRetriever] = None

    def render(self, source: str, src_lang_name: str, trg_lang_name: str, pool_index: Optional[int] = None) -> str:
        if self.config.num_examples <= 0:
            return ""
        retriever = self._get_retriever()
        examples = (
            retriever.retrieve_for_pool_index(pool_index, self.config.num_examples)
            if pool_index is not None
            else retriever.retrieve(source, self.config.num_examples)
        )
        return self.config.formatter.format(examples, src_lang_name, trg_lang_name)

    def _get_retriever(self) -> ExampleRetriever:
        # Built on first use rather than in __init__, so num_examples: 0 never touches the corpus.
        if self._retriever is None:
            self._retriever = self._build_retriever()
        return self._retriever

    def _build_retriever(self) -> ExampleRetriever:
        pairs = read_parallel_text_pairs(self._pool_src_path, self._pool_trg_path)
        if pairs is None:
            raise RuntimeError(
                f"prompt.num_examples > 0 requires the training corpus at {self._pool_src_path} and "
                f"{self._pool_trg_path}. Run preprocessing (--preprocess) first."
            )
        sources, targets = pairs
        retriever = create_example_retriever(self.config.selection_method, self.config.selection_model)
        retriever.fit([Example(source=s, target=t) for s, t in zip(sources, targets)])
        return retriever
