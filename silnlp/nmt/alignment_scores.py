import logging
from pathlib import Path

import pandas as pd

from ..alignment.config import get_aligner_name
from ..alignment.utils import add_alignment_scores
from ..common.corpus import filter_parallel_corpus

LOGGER = logging.getLogger(__name__)


class AlignmentScores:
    """How well each sentence pair of a corpus aligns, cached in the experiment directory."""

    def __init__(self, exp_dir: Path, aligner_id: str, force: bool) -> None:
        self._exp_dir = exp_dir
        self._aligner_id = aligner_id
        self._force = force

    def add_to(self, corpus: pd.DataFrame, source_name: str, target_name: str) -> None:
        cache_path = self._exp_dir / f"{source_name}_{target_name}.csv"
        if cache_path.is_file() and not self._force:
            LOGGER.info(f"Using pre-existing alignment scores from {cache_path}")
            self._apply_cached(corpus, cache_path)
            return
        LOGGER.info(f"Computing alignment scores using {get_aligner_name(self._aligner_id)}")
        add_alignment_scores(corpus, self._aligner_id)
        corpus.to_csv(cache_path, index=False)

    def filter(self, corpus: pd.DataFrame, threshold: float) -> pd.DataFrame:
        unfiltered_count = len(corpus)
        filtered = filter_parallel_corpus(corpus, threshold)
        LOGGER.info(f"Filtered out {unfiltered_count - len(filtered)} pairs with alignment below {threshold}.")
        return filtered.drop("score", axis=1, errors="ignore")

    def _apply_cached(self, corpus: pd.DataFrame, cache_path: Path) -> None:
        # The cached rows are matched to the corpus by position, not by key.
        cached = pd.read_csv(cache_path)
        cached["idx"] = corpus.index
        cached.set_index("idx", inplace=True)
        corpus["score"] = cached["score"]
