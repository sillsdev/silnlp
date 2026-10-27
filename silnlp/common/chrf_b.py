import logging
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import sacrebleu

from .corpus import Term
from .hypothesis_coverage import HypothesisCoverage
from .key_term_consistency import KeyTermConsistency

LOGGER = logging.getLogger(__name__)

# Every Bible checked is at most 10 characters per space-separated word or at least 19.
_MAX_CHARACTERS_PER_WORD = 15.0


class ChrFB:
    """chrF-B ("chirf-bee", B for Bible): chrF3 plus coverage of the hypothesis and key-term consistency, in chrF3
    points. It is for comparing systems on held-out Bible books, not for scoring single verses or general-domain text.

    normalized_corpus_score rescales the chrF3 part between a floor and 100 so scores read alike across languages. The
    floor is chrF3 of the references shuffled among themselves: what text in the right language but with the wrong
    meaning gets for shared names, punctuation and common words (4 to 26 points across Bible test sets). Within one
    test set the floor is a constant, so the two scores always rank systems the same way.

    chrF-B is only validated on scripts that put spaces between words. Thai, Lao, Khmer, Burmese, Tibetan, Shan,
    Chinese and Japanese Bibles space phrases instead, so key terms almost never match and coverage aligns whole
    phrases; scoring them logs a warning.
    """

    # Coverage rises when unaligned words are dropped, so a larger weight would let it excuse omissions.
    _COVERAGE_WEIGHT = 60.0
    _KEY_TERMS_WEIGHT = 20.0
    _FLOOR_SHUFFLES = 3

    def __init__(
        self, coverage: Optional[HypothesisCoverage], key_terms: Optional[KeyTermConsistency], language: str = ""
    ) -> None:
        self._coverage = coverage
        self._key_terms = key_terms
        self._language = language
        self._checked_spacing = False

    @classmethod
    def train(
        cls, corpus: pd.DataFrame, english_corpus: pd.DataFrame, terms: Iterable[Term], language: str = ""
    ) -> "ChrFB":
        """corpus: the source and target Scripture, and english_corpus: English as the source, as from
        get_scripture_parallel_corpus."""
        coverage = HypothesisCoverage.train(corpus["source"].tolist(), corpus["target"].tolist())
        key_terms = KeyTermConsistency.infer(
            english_corpus["vref"].tolist(),
            english_corpus["source"].tolist(),
            english_corpus["target"].tolist(),
            terms,
        )
        return cls(coverage, key_terms, language)

    def corpus_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: List[List[str]]) -> float:
        self._check_spacing(references)
        return _chrf3(hypotheses, references) + self._added_points(sources, hypotheses, references)

    def normalized_corpus_score(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: List[List[str]]
    ) -> float:
        self._check_spacing(references)
        floor = self.floor(references)
        return 100 * (_chrf3(hypotheses, references) - floor) / (100 - floor) + self._added_points(
            sources, hypotheses, references
        )

    @classmethod
    def floor(cls, references: List[List[str]]) -> float:
        """chrF3 of the first reference's lines rotated, so none stays in place, averaged over a few rotations."""
        lines = references[0]
        if len(lines) < 2:
            return 0.0
        rng = np.random.default_rng(0)
        scores = []
        for _ in range(cls._FLOOR_SHUFFLES):
            shift = 1 + int(rng.integers(len(lines) - 1))
            scores.append(_chrf3(lines[shift:] + lines[:shift], references))
        return float(np.mean(scores))

    def _check_spacing(self, references: List[List[str]]) -> None:
        # Once per language pair, so scoring every checkpoint doesn't repeat it.
        if self._checked_spacing:
            return
        self._checked_spacing = True
        if lacks_word_spaces([line for reference in references for line in reference]):
            language = f" for {self._language}" if self._language else ""
            LOGGER.warning(
                f"chrF-B{language} is unreliable: the references have no word breaks, and chrF-B is only validated "
                "on scripts that have them. chrF3 is the score to use."
            )

    def _added_points(self, sources: Sequence[str], hypotheses: Sequence[str], references: List[List[str]]) -> float:
        return self._COVERAGE_WEIGHT * self._mean_over_references(
            self._coverage_scores, sources, hypotheses, references
        ) + self._KEY_TERMS_WEIGHT * self._mean_over_references(self._key_term_scores, sources, hypotheses, references)

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


def lacks_word_spaces(texts: Sequence[str]) -> bool:
    """Whether the texts' script doesn't put spaces between words, judged by the characters between spaces."""
    words = sum(len(text.split()) for text in texts)
    characters = sum(len("".join(text.split())) for text in texts)
    return words > 0 and characters / words > _MAX_CHARACTERS_PER_WORD


def _chrf3(hypotheses: Sequence[str], references: List[List[str]]) -> float:
    return sacrebleu.corpus_chrf(list(hypotheses), references, char_order=6, beta=3, remove_whitespace=True).score
