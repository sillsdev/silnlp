"""Translation-scoring service (the *Evaluate* demo).

Wraps the shared :class:`NllbModel` with :class:`TranslationScorer` to flag
low-probability spans and propose alternatives; it never loads its own model.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

from .model import NllbModel

LOGGER = logging.getLogger(__name__)

DEFAULT_LOW_PROB_THRESHOLD = -3.0
DEFAULT_TOP_K_SUGGESTIONS = 5


@dataclass(frozen=True)
class _FlagCandidate:
    text: str
    span_start: int
    span_end: int
    mean_token_log_prob: float
    suggestions: List[Dict[str, float | str]]
    kind: str

    @property
    def length(self) -> int:
        return self.span_end - self.span_start


def _word_char_spans(text: str) -> List[Tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _build_suggestions(score: Any) -> List[Dict[str, float | str]]:
    return [
        {
            "phrase": suggestion.phrase,
            "mean_token_log_prob": suggestion.mean_token_log_prob,
            "improvement": suggestion.improvement,
        }
        for suggestion in score.suggestions
    ]


def _collect_flag_candidates(scored: Any) -> List[_FlagCandidate]:
    candidates: List[_FlagCandidate] = []
    for score in scored.flagged_phrases:
        candidates.append(
            _FlagCandidate(
                text=score.text,
                span_start=score.word_start,
                span_end=score.word_end,
                mean_token_log_prob=score.mean_token_log_prob,
                suggestions=_build_suggestions(score),
                kind="phrase",
            )
        )

    for score in scored.flagged_words:
        candidates.append(
            _FlagCandidate(
                text=score.text,
                span_start=score.word_start,
                span_end=score.word_end,
                mean_token_log_prob=score.mean_token_log_prob,
                suggestions=_build_suggestions(score),
                kind="word",
            )
        )

    return candidates


def _candidate_weight(candidate: _FlagCandidate) -> float:
    """Return improvement * word_length for the top suggestion, or 0 if none."""
    if not candidate.suggestions:
        return 0.0
    return max(0.0, float(candidate.suggestions[0].get("improvement", 0.0)) * candidate.length)


def _select_max_improvement_flags(candidates: Sequence[_FlagCandidate]) -> List[_FlagCandidate]:
    """Select a non-overlapping subset of flag candidates that maximises
    total_improvement = Σ (top_suggestion.improvement × span_word_length).

    Uses the standard weighted interval scheduling DP (sort by end position,
    binary-search for the latest compatible predecessor, then backtrack).
    Candidates with zero weight are excluded — they cannot improve the objective
    but can block candidates that do.
    """
    items = [(c, _candidate_weight(c)) for c in candidates if _candidate_weight(c) > 0]
    if not items:
        return []

    items.sort(key=lambda x: (x[0].span_end, x[0].span_start))
    n = len(items)

    def last_compatible(i: int) -> int:
        """Largest j < i such that items[j].span_end <= items[i].span_start."""
        target = items[i][0].span_start
        lo, hi, result = 0, i - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if items[mid][0].span_end <= target:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    # dp[i] = max total improvement from a subset of items[0..i-1]  (1-indexed)
    dp = [0.0] * (n + 1)
    for i in range(1, n + 1):
        _, w = items[i - 1]
        p = last_compatible(i - 1)
        dp[i] = max(dp[i - 1], w + dp[p + 1])

    # Backtrack: at each step re-evaluate the same include/skip condition.
    selected: List[_FlagCandidate] = []
    i = n
    while i > 0:
        c, w = items[i - 1]
        p = last_compatible(i - 1)
        if w + dp[p + 1] >= dp[i - 1]:  # including item i is at least as good
            selected.append(c)
            i = p + 1
        else:
            i -= 1

    selected.reverse()
    return selected


def _format_flags(scored: Any) -> List[Dict[str, Any]]:
    char_spans = _word_char_spans(scored.translation)
    flags: List[Dict[str, Any]] = []

    for index, candidate in enumerate(_select_max_improvement_flags(_collect_flag_candidates(scored))):
        if candidate.span_end <= 0 or candidate.span_end > len(char_spans):
            continue
        char_start = char_spans[candidate.span_start][0]
        char_end = char_spans[candidate.span_end - 1][1]
        flags.append(
            {
                "id": f"flag-{index}",
                "text": candidate.text,
                "kind": candidate.kind,
                "span_start": candidate.span_start,
                "span_end": candidate.span_end,
                "char_start": char_start,
                "char_end": char_end,
                "mean_token_log_prob": candidate.mean_token_log_prob,
                "suggestions": candidate.suggestions,
            }
        )

    return flags


class ScoringService:
    """Score a translation and flag low-probability spans with alternatives."""

    def __init__(
        self,
        model: NllbModel,
        low_prob_threshold: float = DEFAULT_LOW_PROB_THRESHOLD,
        top_k_suggestions: int = DEFAULT_TOP_K_SUGGESTIONS,
    ) -> None:
        self._model = model
        self._low_prob_threshold = low_prob_threshold
        self._top_k_suggestions = top_k_suggestions

    def _configure_languages(self, source_lang: str, target_lang: str) -> None:
        tokenizer = self._model.tokenizer
        model = self._model.model

        def _lang_id(lang: str) -> int:
            return tokenizer.convert_tokens_to_ids(lang)

        if _lang_id(source_lang) == tokenizer.unk_token_id:
            raise ValueError(f"Unsupported source language: {source_lang}")
        if _lang_id(target_lang) == tokenizer.unk_token_id:
            raise ValueError(f"Unsupported target language: {target_lang}")

        tokenizer.src_lang = source_lang
        tokenizer.tgt_lang = target_lang
        forced_bos_token_id = _lang_id(target_lang)
        model.config.forced_bos_token_id = forced_bos_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.forced_bos_token_id = forced_bos_token_id

    def score(self, source: str, translation: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        source = source.strip()
        translation = translation.strip()
        if source == "" or translation == "":
            return {"source": source, "translation": translation, "flags": []}

        from ..translation_scorer import AbsoluteThresholdAnomalyDetector, TranslationScorer

        with self._model.lock:
            self._configure_languages(source_lang, target_lang)
            scorer = TranslationScorer(
                self._model.model,
                self._model.tokenizer,
                anomaly_detector=AbsoluteThresholdAnomalyDetector(self._low_prob_threshold),
                top_k_suggestions=self._top_k_suggestions,
            )
            t0 = time.perf_counter()
            scored = scorer.score(source, translation)
            LOGGER.info("score() completed in %.2f s", time.perf_counter() - t0)
            if self._model.device.type == "cuda":
                LOGGER.info(
                    "VRAM allocated: %.0f MiB, reserved: %.0f MiB",
                    torch.cuda.memory_allocated(self._model.device) / 1024**2,
                    torch.cuda.memory_reserved(self._model.device) / 1024**2,
                )
        return {
            "source": source,
            "translation": scored.translation,
            "flags": _format_flags(scored),
        }
