from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import sacrebleu

from .corpus import Term
from .hypothesis_coverage import HypothesisCoverage
from .key_term_consistency import KeyTermConsistency


class ChrFB:
    """chrF-B ("chirf-bee", B for Bible): chrF3 plus coverage of the hypothesis and key-term consistency, in chrF3
    points. It is for comparing systems on held-out Bible books, not for scoring single verses or general-domain text.
    """

    # Coverage rises when unaligned words are dropped, so a larger weight would let it excuse omissions.
    _COVERAGE_WEIGHT = 60.0
    _KEY_TERMS_WEIGHT = 20.0

    def __init__(self, coverage: Optional[HypothesisCoverage], key_terms: Optional[KeyTermConsistency]) -> None:
        self._coverage = coverage
        self._key_terms = key_terms

    @classmethod
    def train(cls, corpus: pd.DataFrame, english_corpus: pd.DataFrame, terms: Iterable[Term]) -> "ChrFB":
        """corpus: the source and target Scripture, and english_corpus: English as the source, as from
        get_scripture_parallel_corpus."""
        coverage = HypothesisCoverage.train(corpus["source"].tolist(), corpus["target"].tolist())
        key_terms = KeyTermConsistency.infer(
            english_corpus["vref"].tolist(),
            english_corpus["source"].tolist(),
            english_corpus["target"].tolist(),
            terms,
        )
        return cls(coverage, key_terms)

    def corpus_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: List[List[str]]) -> float:
        chrf3 = sacrebleu.corpus_chrf(list(hypotheses), references, char_order=6, beta=3, remove_whitespace=True)
        return (
            chrf3.score
            + self._COVERAGE_WEIGHT * self._mean_over_references(self._coverage_scores, sources, hypotheses, references)
            + self._KEY_TERMS_WEIGHT
            * self._mean_over_references(self._key_term_scores, sources, hypotheses, references)
        )

    def _mean_over_references(
        self,
        scorer: Callable[[List[str], List[str], List[str]], np.ndarray],
        sources: Sequence[str],
        hypotheses: Sequence[str],
        references: List[List[str]],
    ) -> float:
        # A part that can't be scored (no aligner, no key terms in the references) adds nothing, so chrF3 decides.
        scores = []
        for reference in references:
            kept = [i for i, line in enumerate(reference) if line]
            if kept:
                scores.append(
                    scorer([sources[i] for i in kept], [hypotheses[i] for i in kept], [reference[i] for i in kept])
                )
        all_scores = np.concatenate(scores) if scores else np.array([])
        return float(np.nanmean(all_scores)) if np.any(~np.isnan(all_scores)) else 0.0

    def _coverage_scores(self, sources: List[str], hypotheses: List[str], references: List[str]) -> np.ndarray:
        if self._coverage is None:
            return np.full(len(hypotheses), np.nan)
        return self._coverage.verse_scores(sources, hypotheses, references)

    def close(self) -> None:
        if self._coverage is not None:
            self._coverage.close()

    def _key_term_scores(self, sources: List[str], hypotheses: List[str], references: List[str]) -> np.ndarray:
        if self._key_terms is None:
            return np.full(len(hypotheses), np.nan)
        return self._key_terms.verse_scores(hypotheses, references)
