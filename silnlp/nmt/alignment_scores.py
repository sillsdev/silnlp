import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from ..alignment.config import get_aligner_name
from ..alignment.utils import add_alignment_scores
from ..common.corpus import filter_parallel_corpus

LOGGER = logging.getLogger(__name__)


class CorpusAligner(ABC):
    @abstractmethod
    def add_scores(self, corpus: pd.DataFrame, aligner_id: str) -> None:
        ...


class ToolCorpusAligner(CorpusAligner):
    def add_scores(self, corpus: pd.DataFrame, aligner_id: str) -> None:
        LOGGER.info(f"Computing alignment scores using {get_aligner_name(aligner_id)}")
        add_alignment_scores(corpus, aligner_id)


class AlignmentScores:
    """How well each verse pair of a corpus aligns, cached in the experiment directory."""

    def __init__(self, exp_dir: Path, aligner_id: str, force: bool, aligner: Optional[CorpusAligner] = None) -> None:
        self._exp_dir = exp_dir
        self._aligner_id = aligner_id
        self._force = force
        self._aligner = aligner if aligner is not None else ToolCorpusAligner()

    def add_to(self, corpus: pd.DataFrame, source_name: str, target_name: str) -> None:
        cache_path = self._exp_dir / f"{source_name}_{target_name}.csv"
        if cache_path.is_file() and not self._force:
            LOGGER.info(f"Using pre-existing alignment scores from {cache_path}")
            self._apply_cached(corpus, cache_path)
            return
        self._aligner.add_scores(corpus, self._aligner_id)
        corpus.to_csv(cache_path, index=False)

    def filter(self, corpus: pd.DataFrame, threshold: float) -> pd.DataFrame:
        unfiltered_count = len(corpus)
        filtered = filter_parallel_corpus(corpus, threshold)
        LOGGER.info(
            f"Filtered out {unfiltered_count - len(filtered)} verses pairs with alignment below {threshold}."
        )
        return filtered.drop("score", axis=1, errors="ignore")

    def _apply_cached(self, corpus: pd.DataFrame, cache_path: Path) -> None:
        # The cache holds no verse references, so its rows are matched to the corpus by position.
        cached = pd.read_csv(cache_path)
        cached["idx"] = corpus.index
        cached.set_index("idx", inplace=True)
        corpus["score"] = cached["score"]
