"""Retrieval of in-context translation examples from a parallel corpus.

Used by the in-context learning model (:mod:`silnlp.nmt.icl_config`) to select the most
relevant source/target pairs from the training corpus for each translation request. Two
lexical scoring methods are supported, selected with ``infer.retrieval.method``:

* ``tfidf`` - TF-IDF vectors with cosine similarity, via scikit-learn.
* ``bm25`` - Okapi BM25, via the optional ``rank_bm25`` package.

Both index the *source* side of the corpus and are purely lexical, which suits the
low-resource languages this pipeline targets: no pretrained embedding model is needed, and
none is reliably available for them anyway.
"""

import json
import logging
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

LOGGER = logging.getLogger(__name__ + ".example_retrieval")

BM25_METHOD = "bm25"
TFIDF_METHOD = "tfidf"
VALID_RETRIEVAL_METHODS = (BM25_METHOD, TFIDF_METHOD)

RETRIEVER_FILENAME = "retrieval.pkl"
RETRIEVER_META_FILENAME = "retrieval_meta.json"

# Shared by both methods so that switching between them doesn't also change tokenization.
_TOKEN_PATTERN = r"\w+"


def tokenize_for_retrieval(text: str) -> List[str]:
    return re.findall(_TOKEN_PATTERN, text.lower())


@dataclass(frozen=True)
class ExamplePair:
    source: str
    target: str


class ExampleRetriever(ABC):
    """A fitted index over the source side of a parallel corpus."""

    method: str = ""

    def __init__(self) -> None:
        self._pairs: List[ExamplePair] = []

    @property
    def pairs(self) -> List[ExamplePair]:
        return self._pairs

    def fit(self, pairs: Sequence[ExamplePair]) -> None:
        self._pairs = list(pairs)
        self._fit_index([pair.source for pair in self._pairs])

    def retrieve(self, query: str, k: int) -> List[ExamplePair]:
        """Return up to ``k`` example pairs, most relevant first."""
        if k <= 0 or len(self._pairs) == 0:
            return []
        ranked_indices = self._rank(query, min(k, len(self._pairs)))
        return [self._pairs[i] for i in ranked_indices]

    @abstractmethod
    def _fit_index(self, sources: List[str]) -> None:
        ...

    @abstractmethod
    def _rank(self, query: str, k: int) -> List[int]:
        """Return the indices of the ``k`` best-matching sources, most relevant first."""

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / RETRIEVER_FILENAME).open("wb") as file:
            pickle.dump(self, file)
        meta = {"method": self.method, "num_pairs": len(self._pairs)}
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
        # TF-IDF vectors are L2-normalized by default, so a linear kernel is cosine similarity.
        self._vectorizer = TfidfVectorizer(lowercase=True, token_pattern=_TOKEN_PATTERN)
        self._matrix = self._vectorizer.fit_transform(sources)

    def _rank(self, query: str, k: int) -> List[int]:
        from sklearn.metrics.pairwise import linear_kernel

        if self._vectorizer is None or self._matrix is None:
            return []
        query_vector = self._vectorizer.transform([query])
        scores = linear_kernel(query_vector, self._matrix)[0]
        return [int(i) for i in scores.argsort()[-k:][::-1]]


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

    def _rank(self, query: str, k: int) -> List[int]:
        if self._index is None:
            return []
        scores = self._index.get_scores(tokenize_for_retrieval(query))
        return [int(i) for i in scores.argsort()[-k:][::-1]]


def _import_bm25():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError(
            "BM25 example retrieval requires the 'rank_bm25' package, which is part of the 'icl' "
            "extra. Install it with `poetry install -E icl`, or set infer.retrieval.method to "
            "'tfidf' to use the built-in TF-IDF retriever instead."
        ) from e
    return BM25Okapi


def create_retriever(method: str) -> ExampleRetriever:
    normalized = method.lower()
    if normalized == TFIDF_METHOD:
        return TfidfExampleRetriever()
    if normalized == BM25_METHOD:
        return BM25ExampleRetriever()
    raise ValueError(f"Unknown retrieval method '{method}'. Valid options: {', '.join(VALID_RETRIEVAL_METHODS)}.")
