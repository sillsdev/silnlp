"""Few-shot examples for LLM translation prompts: retrieval (ExampleRetriever), formatting
(ExampleFormatter), and prompt assembly (PromptExampleConfig, ExamplePromptBuilder)."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from .corpora import read_parallel_text_pairs

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Example:
    source: str
    target: str


# TranslateGemma's chat template requires structured {type, lang_code, text} content instead of
# free text, so it cannot carry few-shot examples.
TRANSLATE_GEMMA_MODEL_PREFIXES = ("google/translate-gemma", "google/translategemma")


class _EmbeddingModel(Protocol):
    """SentenceTransformer's interface, narrowed to what's used here; also the test injection point."""

    def encode(
        self, texts: Sequence[str], convert_to_numpy: bool, normalize_embeddings: bool, show_progress_bar: bool
    ) -> np.ndarray:
        ...


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
    def __init__(self, examples: Sequence[Example]) -> None:
        self._examples: List[Example] = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def retrieve(self, query: str, k: int) -> List[Example]:
        """Top-k most similar pool examples to an arbitrary query string not in the pool
        (used at eval/test/translate time)."""
        if k <= 0 or len(self._examples) == 0:
            return []
        return [self._examples[i] for i in self._top_indices_for_query(query, k)]

    def retrieve_for_pool_index(self, index: int, k: int) -> List[Example]:
        """Top-k most similar pool examples to the pool entry at `index`, excluding itself
        (leave-one-out; used during training)."""
        if k <= 0 or len(self._examples) == 0:
            return []
        return [self._examples[i] for i in self._top_indices_for_pool_index(index, k)]

    @abstractmethod
    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        ...

    @abstractmethod
    def _top_indices_for_pool_index(self, index: int, k: int) -> List[int]:
        ...


class TfidfExampleRetriever(ExampleRetriever):
    def __init__(self, examples: Sequence[Example]) -> None:
        super().__init__(examples)
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer()
        sources = [ex.source for ex in self._examples]
        # Rows are L2-normalized by default, so a dot product against the matrix is cosine similarity.
        self._matrix = self._vectorizer.fit_transform(sources) if sources else None

    def _scores_for_vector(self, vector) -> np.ndarray:
        return (self._matrix @ vector.T).toarray().ravel()

    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        vector = self._vectorizer.transform([query])
        return _top_k_indices(self._scores_for_vector(vector), k)

    def _top_indices_for_pool_index(self, index: int, k: int) -> List[int]:
        # Guarded by retrieve_for_pool_index()'s empty-pool check, so this is always fit here.
        assert self._matrix is not None
        vector = self._matrix[index]
        return _top_k_indices(self._scores_for_vector(vector), k, exclude=index)


def _load_sentence_transformer(model_name: str) -> _EmbeddingModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "params.prompt.example_selection.method: embedding requires the "
            "'sentence-transformers' package. Install it with `poetry install -E llm`."
        ) from e
    return SentenceTransformer(model_name)


class EmbeddingExampleRetriever(ExampleRetriever):
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, examples: Sequence[Example], model: Optional[_EmbeddingModel] = None) -> None:
        """`model` is the test injection seam; production leaves it unset."""
        super().__init__(examples)
        self._model = model if model is not None else _load_sentence_transformer(self.DEFAULT_MODEL)

        sources = [ex.source for ex in self._examples]
        self._embeddings: np.ndarray = (
            self._model.encode(sources, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            if sources
            else np.zeros((0, 0), dtype=np.float32)
        )

    def _top_indices_for_query(self, query: str, k: int) -> List[int]:
        vector = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[
            0
        ]
        return _top_k_indices(self._embeddings @ vector, k)

    def _top_indices_for_pool_index(self, index: int, k: int) -> List[int]:
        vector = self._embeddings[index]
        return _top_k_indices(self._embeddings @ vector, k, exclude=index)


def create_example_retriever(
    method: str, examples: Sequence[Example], model_name: Optional[str] = None
) -> ExampleRetriever:
    if method == "lexical":
        return TfidfExampleRetriever(examples)
    if method == "embedding":
        model = _load_sentence_transformer(model_name) if model_name else None
        return EmbeddingExampleRetriever(examples, model=model)
    raise ValueError(f"Unknown example_selection.method '{method}'. Valid options: lexical, embedding.")


class ExampleFormatter(ABC):
    """Renders retrieved examples into the text that fills {examples} in instruction_template."""

    @abstractmethod
    def format(self, examples: Sequence[Example], src_lang_name: str, trg_lang_name: str) -> str:
        ...


class TextExampleFormatter(ExampleFormatter):
    """Unlike JsonExampleFormatter/XmlExampleFormatter, does not escape the template output."""

    def __init__(self, template: str) -> None:
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


def create_example_formatter(format_params: dict) -> ExampleFormatter:
    format_type = str(format_params.get("type", "text")).lower()
    if format_type == "text":
        return TextExampleFormatter(format_params["template"])
    if format_type == "json":
        return JsonExampleFormatter()
    if format_type == "xml":
        return XmlExampleFormatter()
    raise ValueError(f"Unknown params.prompt.example_format.type '{format_type}'. Valid options: text, json, xml.")


class PromptExampleConfig:
    """Parsed params.prompt config for few-shot examples."""

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
            raise ValueError(f"params.prompt.num_examples must be non-negative, got {num_examples}.")

        selection_method = selection_method.lower()
        if selection_method not in ("lexical", "embedding"):
            raise ValueError(
                f"Unknown params.prompt.example_selection.method '{selection_method}'. "
                "Valid options: lexical, embedding."
            )

        if num_examples > 0:
            if "{examples}" not in instruction_template:
                LOGGER.warning(
                    "params.prompt.num_examples > 0 requires '{examples}' in params.prompt.instruction_template, "
                    "otherwise the retrieved examples are silently discarded."
                )
            if model.lower().startswith(TRANSLATE_GEMMA_MODEL_PREFIXES):
                raise RuntimeError(
                    "TranslateGemma models do not support few-shot examples in the prompt. Set params.prompt.num_examples to 0 or use a different model."
                )

        self.num_examples = num_examples
        self.formatter = formatter
        self.selection_method = selection_method
        self.selection_model = selection_model

    @staticmethod
    def from_params(prompt_params: dict, model: str) -> "PromptExampleConfig":
        return PromptExampleConfig(
            num_examples=int(prompt_params["num_examples"]),
            formatter=create_example_formatter(prompt_params["example_format"]),
            selection_method=str(prompt_params["example_selection"]["method"]),
            selection_model=prompt_params["example_selection"].get("model"),
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
                f"params.prompt.num_examples > 0 requires the training corpus at {self._pool_src_path} and "
                f"{self._pool_trg_path}. Run preprocessing (--preprocess) first."
            )
        sources, targets = pairs
        examples = [Example(source=s, target=t) for s, t in zip(sources, targets)]
        return create_example_retriever(self.config.selection_method, examples, model_name=self.config.selection_model)
