"""Few-shot examples for LLM translation prompts: the example corpus (ExamplePool), retrieval
(ExampleRetriever) and formatting (ExampleFormatter)."""

import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Union
from xml.sax.saxutils import escape as xml_escape

import numpy as np
from machine.tokenization import LatinWordTokenizer

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
# Its per-call state is local, so one instance is safe to share across threads.
_WORD_TOKENIZER = LatinWordTokenizer()

# Certain LLMs like Translate Gemma do not support arbitrary prompts
FIXED_PROMPT_MODEL_PREFIXES = ("google/translate-gemma", "google/translategemma")


def tokenize_for_retrieval(text: str) -> List[str]:
    # Punctuation tokens carry no retrieval signal and skew BM25, which scores against length.
    return [token for token in _WORD_TOKENIZER.tokenize(text.lower()) if any(c.isalnum() for c in token)]


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

        # TfidfVectorizer rejects a corpus with nothing to put in its vocabulary.
        if not any(tokenize_for_retrieval(source) for source in sources):
            self._vectorizer = None
            self._matrix = None
            return
        # token_pattern=None keeps scikit-learn from warning that it is unused.
        self._vectorizer = TfidfVectorizer(lowercase=False, tokenizer=tokenize_for_retrieval, token_pattern=None)
        # Rows are L2-normalized by default, so a dot product against the matrix is cosine similarity.
        self._matrix = self._vectorizer.fit_transform(sources)

    def _scores_for_vector(self, vector) -> np.ndarray:
        return (self._matrix @ vector.T).toarray().ravel()

    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        if self._vectorizer is None or self._matrix is None:
            return []
        return _top_k_indices(self._scores_for_vector(self._vectorizer.transform([query])), k)

    def _top_indices_for_pool_index(self, index: int, k: int) -> List[int]:
        if self._matrix is None:
            return []
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


class ExamplePool:
    """The parallel corpus that few-shot examples are drawn from, and its retrieval index."""

    def __init__(
        self, corpus_paths: Sequence[Tuple[Path, Path]], method: str, model_name: Optional[str] = None
    ) -> None:
        self._corpus_paths = list(corpus_paths)
        self._method = method
        self._model_name = model_name
        self._examples: Optional[List[Example]] = None
        self._retriever: Optional[ExampleRetriever] = None

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def method(self) -> str:
        return self._method

    @property
    def examples(self) -> List[Example]:
        # Read on first use rather than in __init__, so num_examples: 0 never touches the corpus.
        if self._examples is None:
            pairs = self._read_first_available_corpus()
            if pairs is None:
                raise RuntimeError(
                    f"num_examples > 0 requires the training corpus at {self._describe_corpus_paths()}. "
                    "Run preprocessing (--preprocess) first."
                )
            self._examples = [Example(source=s, target=t) for s, t in zip(*pairs)]
        return self._examples

    def _read_first_available_corpus(self) -> Optional[Tuple[List[str], List[str]]]:
        for src_path, trg_path in self._corpus_paths:
            pairs = read_parallel_text_pairs(src_path, trg_path)
            if pairs is not None:
                return pairs
        return None

    def _describe_corpus_paths(self) -> str:
        return " or ".join(f"{src_path} and {trg_path}" for src_path, trg_path in self._corpus_paths)

    def ensure_available(self) -> None:
        """Read the corpus now, so a missing one is reported before an expensive step starts."""
        self.examples

    def covers_whole_pool(self, k: int) -> bool:
        return k > 0 and k >= len(self)

    def select(self, query: str, k: int, pool_index: Optional[int] = None) -> List[Example]:
        """Most relevant last, so the best examples sit nearest the source text. Excluding
        `pool_index` keeps the entry being translated from leaking its own target into the prompt."""
        if k <= 0 or len(self) == 0:
            return []
        if self.covers_whole_pool(k):
            # Every example fits, so keep them in corpus order and never build an index.
            return [ex for i, ex in enumerate(self.examples) if i != pool_index]
        retriever = self.get_retriever()
        ranked = (
            retriever.retrieve_for_pool_index(pool_index, k) if pool_index is not None else retriever.retrieve(query, k)
        )
        return list(reversed(ranked))

    def get_retriever(self) -> ExampleRetriever:
        if self._retriever is None:
            retriever = create_example_retriever(self._method, self._model_name)
            retriever.fit(self.examples)
            self._retriever = retriever
        return self._retriever

    def save_index(self, directory: Path) -> None:
        self.get_retriever().save(directory)

    def load_index(self, directory: Path) -> bool:
        """Adopt a previously saved index, or report that one has to be built."""
        retriever = ExampleRetriever.load(directory)
        if retriever is None:
            return False
        if retriever.method != self._method or retriever.model_name != self._model_name:
            LOGGER.info(
                "The saved retrieval index uses '%s' but the config asks for '%s'; rebuilding it.",
                retriever.method,
                self._method,
            )
            return False
        self._retriever = retriever
        self._examples = retriever.examples
        return True
