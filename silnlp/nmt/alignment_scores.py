import logging
from pathlib import Path
from typing import Optional

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
        cached_scores = None if self._force else self._read_cached_scores(cache_path, len(corpus))
        if cached_scores is not None:
            LOGGER.info(f"Using pre-existing alignment scores from {cache_path}")
            # The cached rows are matched to the corpus by position, not by key.
            corpus["score"] = cached_scores.to_numpy()
            return
        LOGGER.info(f"Computing alignment scores using {get_aligner_name(self._aligner_id)}")
        add_alignment_scores(corpus, self._aligner_id)
        corpus.to_csv(cache_path, index=False)

    def filter(self, corpus: pd.DataFrame, threshold: float) -> pd.DataFrame:
        unfiltered_count = len(corpus)
        filtered = filter_parallel_corpus(corpus, threshold)
        LOGGER.info(f"Filtered out {unfiltered_count - len(filtered)} pairs with alignment below {threshold}.")
        return filtered.drop("score", axis=1, errors="ignore")

    def _read_cached_scores(self, cache_path: Path, corpus_size: int) -> Optional[pd.Series]:
        if not cache_path.is_file():
            return None
        scores = pd.read_csv(cache_path, usecols=["score"])["score"]
        if len(scores) != corpus_size:
            LOGGER.warning(f"Ignoring {cache_path}, which has {len(scores)} scores for {corpus_size} pairs.")
            return None
        return scores
