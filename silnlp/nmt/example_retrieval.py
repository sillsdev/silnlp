"""Few-shot examples for LLM translation prompts: the example corpus (ExamplePool), retrieval
(ExampleRetriever) and formatting (ExampleFormatter)."""

import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Tuple, Union
from xml.sax.saxutils import escape as xml_escape

import numpy as np
from machine.tokenization import LatinWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpora import read_parallel_text_pairs

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Example:
    source: str
    target: str


class _EmbeddingModel(Protocol):
    """SentenceTransformer's interface, narrowed to what's used here; also the test injection point."""

    def encode(
        self, texts: Sequence[str], convert_to_numpy: bool, normalize_embeddings: bool, show_progress_bar: bool
    ) -> np.ndarray: ...


class ExampleRetriever(ABC):

    method: str = ""

    def __init__(self) -> None:
        self._source_count = 0

    def _top_indices(self, scores: np.ndarray, k: int, exclude: Optional[int] = None) -> List[int]:
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

    def fit(self, sources: Sequence[str]) -> None:
        self._source_count = len(sources)
        self._fit_index(list(sources))

    def rank(self, query: str, k: int) -> List[int]:
        """Positions of the k best-matching sources, most relevant first."""
        if k <= 0 or self._source_count == 0:
            return []
        return self._top_indices_for_query(query, k)

    def rank_excluding(self, source: str, index: int, k: int) -> List[int]:
        """Ranks `source`, which is the entry at `index`, against every entry but itself."""
        if k <= 0 or self._source_count == 0:
            return []
        return self._top_indices_excluding(source, index, k)

    @abstractmethod
    def _fit_index(self, sources: List[str]) -> None: ...

    @abstractmethod
    def _top_indices_for_query(self, query: str, k: int) -> List[int]: ...

    def _top_indices_excluding(self, source: str, index: int, k: int) -> List[int]:
        # Asks for k + 1 so that dropping the entry itself still leaves k results.
        return [i for i in self._top_indices_for_query(source, k + 1) if i != index][:k]

    def save(self, directory: Path) -> None:
        pass

    def load(self, directory: Path, corpus_size: int) -> bool:
        return False


class RetrievalTokenizer:
    """Splits text into the words a lexical index scores against."""

    def __init__(self) -> None:
        self._tokenizer = LatinWordTokenizer()

    def tokenize(self, text: str) -> List[str]:
        return self._keep_words(self._tokenizer.tokenize(text.lower()))

    def _keep_words(self, tokens: Iterable[str]) -> List[str]:
        # Punctuation tokens carry no retrieval signal and skew BM25, which scores against length.
        return [token for token in tokens if any(c.isalnum() for c in token)]


class LexicalExampleRetriever(ExampleRetriever):
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer = RetrievalTokenizer()

    # Lexical indices can be rebuilt in seconds, so they do not implement save/load.


class TfidfExampleRetriever(LexicalExampleRetriever):
    method = "tfidf"

    def __init__(self) -> None:
        super().__init__()
        self._vectorizer: Optional[Any] = None
        self._matrix: Optional[Any] = None

    def _fit_index(self, sources: List[str]) -> None:
        # TfidfVectorizer rejects a corpus with nothing to put in its vocabulary.
        if not any(self._tokenizer.tokenize(source) for source in sources):
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(lowercase=False, tokenizer=self._tokenizer.tokenize, token_pattern=None)
        self._matrix = self._vectorizer.fit_transform(sources)

    def _scores_for_vector(self, vector) -> np.ndarray:
        return (self._matrix @ vector.T).toarray().ravel()

    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        if self._vectorizer is None or self._matrix is None:
            return []
        return self._top_indices(self._scores_for_vector(self._vectorizer.transform([query])), k)

    def _top_indices_excluding(self, source: str, index: int, k: int) -> List[int]:
        if self._matrix is None:
            return []
        return self._top_indices(self._scores_for_vector(self._matrix[index]), k, exclude=index)


class BM25ExampleRetriever(LexicalExampleRetriever):
    method = "bm25"

    def __init__(self) -> None:
        super().__init__()
        self._index: Optional[Any] = None

    def _fit_index(self, sources: List[str]) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [self._tokenizer.tokenize(source) for source in sources]
        # BM25Okapi rejects an empty corpus and divides by zero on all-empty documents.
        if sum(len(tokens) for tokens in tokenized) == 0:
            self._index = None
            return
        self._index = BM25Okapi(tokenized)

    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        if self._index is None:
            return []
        return self._top_indices(self._index.get_scores(self._tokenizer.tokenize(query)), k)


class EmbeddingExampleRetriever(ExampleRetriever):
    method = "embedding"

    _DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _EMBEDDINGS_FILENAME = "example_retrieval_embeddings.npy"
    _META_FILENAME = "example_retrieval_meta.json"

    def __init__(self, model_name: Optional[str] = None, model: Optional[_EmbeddingModel] = None) -> None:
        """`model` is the test injection seam; production passes only `model_name`."""
        super().__init__()
        self._name = model_name or self._DEFAULT_MODEL
        self._model = model
        self._model_lock = threading.Lock()
        self._embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def _get_model(self) -> _EmbeddingModel:
        # Inference ranks on a thread pool, so without this every worker would load its own copy.
        with self._model_lock:
            if self._model is None:
                self._model = self._create_model()
            return self._model

    def _create_model(self) -> _EmbeddingModel:
        # Imported here so that a config using no embedding retrieval never pays for the import.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._name)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._get_model().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    def _fit_index(self, sources: List[str]) -> None:
        self._embeddings = self._encode(sources) if sources else np.zeros((0, 0), dtype=np.float32)

    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        if self._embeddings.shape[0] == 0:
            return []
        return self._top_indices(self._embeddings @ self._encode([query])[0], k)

    def _top_indices_excluding(self, source: str, index: int, k: int) -> List[int]:
        return self._top_indices(self._embeddings @ self._embeddings[index], k, exclude=index)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / self._EMBEDDINGS_FILENAME, self._embeddings)
        meta = {"method": self.method, "model_name": self._name, "num_sources": self._source_count}
        with (directory / self._META_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)

    def load(self, directory: Path, corpus_size: int) -> bool:
        try:
            with (directory / self._META_FILENAME).open("r", encoding="utf-8") as file:
                meta = json.load(file)
            embeddings = np.load(directory / self._EMBEDDINGS_FILENAME, allow_pickle=False)
        except FileNotFoundError:
            return False
        except Exception:
            LOGGER.warning("Could not read the retrieval index in %s; it will be rebuilt.", directory, exc_info=True)
            return False

        reason = self._reason_to_rebuild(meta, len(embeddings), corpus_size)
        if reason is not None:
            LOGGER.info("%s; rebuilding it.", reason)
            return False

        self._embeddings = embeddings
        self._source_count = len(embeddings)
        return True

    def _reason_to_rebuild(self, meta: dict, saved_count: int, corpus_size: int) -> Optional[str]:
        """Why a saved index cannot stand in for this one over a corpus of `corpus_size`."""
        if meta.get("method") != self.method or meta.get("model_name") != self._name:
            return (
                f"The saved retrieval index uses '{meta.get('method')}' with model "
                f"'{meta.get('model_name')}' but the config asks for '{self.method}' with '{self._name}'"
            )
        if saved_count != corpus_size:
            return f"The saved retrieval index covers {saved_count} examples but the corpus has {corpus_size}"
        return None


class ExampleRetrieverFactory:
    """Creates the retriever named by a config's example_selection.method."""

    DEFAULT_METHOD = TfidfExampleRetriever.method

    @classmethod
    def create(cls, method: str, model_name: Optional[str] = None) -> ExampleRetriever:
        """Creates it unfitted; ExamplePool fits it with its own sources."""
        normalized = method.lower()
        if normalized == TfidfExampleRetriever.method:
            return TfidfExampleRetriever()
        if normalized == BM25ExampleRetriever.method:
            return BM25ExampleRetriever()
        if normalized == EmbeddingExampleRetriever.method:
            return EmbeddingExampleRetriever(model_name)
        raise ValueError(f"Unknown example_selection.method '{method}'. Valid options: {cls._method_names()}.")

    @classmethod
    def _method_names(cls) -> str:
        return ", ".join((TfidfExampleRetriever.method, BM25ExampleRetriever.method, EmbeddingExampleRetriever.method))


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
        payload = [{"source": ex.source, "target": ex.target} for ex in examples]
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


class XmlExampleFormatter(ExampleFormatter):
    def format(self, examples: Sequence[Example], src_lang_name: str, trg_lang_name: str) -> str:
        return "".join(
            f"<example>\n<source>{xml_escape(ex.source)}</source>\n<target>{xml_escape(ex.target)}</target>\n"
            "</example>\n"
            for ex in examples
        )


class ExampleFormatterFactory:
    @classmethod
    def create(cls, format_params: Union[str, dict]) -> ExampleFormatter:
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


@dataclass(frozen=True)
class ExamplePoolSummary:
    corpus_size: int
    selection_method: str


@dataclass(frozen=True)
class CorpusPair:
    """The source and target files of one parallel corpus."""

    src_path: Path
    trg_path: Path

    def exists(self) -> bool:
        return self.src_path.is_file() and self.trg_path.is_file()

    def read(self) -> Optional[Tuple[List[str], List[str]]]:
        return read_parallel_text_pairs(self.src_path, self.trg_path)

    def describe(self) -> str:
        return f"{self.src_path} and {self.trg_path}"


class CorpusPairProvider(ABC):
    """Names the corpus to draw examples from, when asked rather than up front, since the files
    are written by a preprocess step that runs after the config is built."""

    @abstractmethod
    def get_corpus_pair(self) -> CorpusPair: ...


class FixedCorpusPairProvider(CorpusPairProvider):
    def __init__(self, corpus_pair: CorpusPair) -> None:
        self._corpus_pair = corpus_pair

    def get_corpus_pair(self) -> CorpusPair:
        return self._corpus_pair


class PreferredCorpusPairProvider(CorpusPairProvider):
    def __init__(self, preferred: CorpusPair, fallback: CorpusPair) -> None:
        self._preferred = preferred
        self._fallback = fallback

    def get_corpus_pair(self) -> CorpusPair:
        return self._preferred if self._preferred.exists() else self._fallback


class ExamplePool:
    """The parallel corpus that few-shot examples are drawn from, and its retrieval index."""

    def __init__(self, corpus_provider: CorpusPairProvider, retriever: ExampleRetriever) -> None:
        self._corpus_provider = corpus_provider
        self._retriever = retriever
        self._examples: Optional[List[Example]] = None
        self._fitted = False
        # Reentrant because fitting the index reads the corpus through the same lock.
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.all_examples())

    def all_examples(self) -> List[Example]:
        # Read on first use rather than in __init__, so num_examples: 0 never touches the corpus.
        with self._lock:
            if self._examples is None:
                corpus_pair = self._corpus_provider.get_corpus_pair()
                lines = corpus_pair.read()
                if lines is None:
                    raise RuntimeError(
                        f"num_examples > 0 requires the training corpus at {corpus_pair.describe()}. "
                        "Run preprocessing (--preprocess) first."
                    )
                self._examples = [Example(source=s, target=t) for s, t in zip(*lines)]
            return self._examples

    def summarize(self) -> "ExamplePoolSummary":
        return ExamplePoolSummary(len(self), self._retriever.method)

    def ensure_available(self) -> None:
        """Read the corpus now, so a missing one is reported before an expensive step starts."""
        self.all_examples()

    def covers_whole_pool(self, k: int) -> bool:
        return k > 0 and k >= len(self)

    def select(self, query: str, k: int, pool_index: Optional[int] = None) -> List[Example]:
        """Selects the k most relevant examples for `query`, optionally excluding the example at `pool_index`.
        If k < pool size, the examples are returned in order of relevance; otherwise, corpus order."""

        if k <= 0 or len(self) == 0:
            return []
        if self.covers_whole_pool(k):
            # Every example fits, so keep them in corpus order and never build an index.
            return [ex for i, ex in enumerate(self.all_examples()) if i != pool_index]
        retriever = self.get_retriever()
        if pool_index is None:
            ranked = retriever.rank(query, k)
        else:
            ranked = retriever.rank_excluding(self.all_examples()[pool_index].source, pool_index, k)
        return [self.all_examples()[i] for i in reversed(ranked)]

    def get_retriever(self) -> ExampleRetriever:
        with self._lock:
            if not self._fitted:
                self._retriever.fit([example.source for example in self.all_examples()])
                self._fitted = True
            return self._retriever

    def save_index(self, directory: Path) -> None:
        self.get_retriever().save(directory)

    def load_index(self, directory: Path) -> bool:
        """Adopt a previously saved index, or report that one has to be built."""
        with self._lock:
            if not self._retriever.load(directory, len(self)):
                return False
            self._fitted = True
            return True
