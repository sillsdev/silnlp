"""Inline translation-suggestion service (the *Suggest* demo).

Wraps the shared :class:`NllbModel` with the production suggestion logic from
:class:`HuggingFaceNMTModel`; it never loads its own model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from ..config import TranslationSuggester
from ..hugging_face_config import (
    HuggingFaceNMTModel,
    HuggingFaceTranslationSuggester,
    SilTranslationPipeline,
)
from .model import NllbModel

LOGGER = logging.getLogger(__name__)

UTF8_ENCODING = "utf-8"


def _ensure_utf8_text(text: str) -> str:
    return text.encode(UTF8_ENCODING, errors="strict").decode(UTF8_ENCODING)


class _NllbSuggesterModelAdapter:
    """Reuse the production suggestion logic from :class:`HuggingFaceNMTModel`."""

    def _split_partial_translation(self, tokenizer: Any, partial_translation: str) -> Tuple[str, str]:
        return HuggingFaceNMTModel._split_partial_translation(self, tokenizer, partial_translation)

    def _build_decoder_input_ids(self, tokenizer: Any, model: Any, prefix: str) -> Any:
        return HuggingFaceNMTModel._build_decoder_input_ids(self, tokenizer, model, prefix)

    def _extract_suggestion_text(
        self,
        tokenizer: Any,
        partial_translation: str,
        translation_token_ids: list[int],
        token_scores: Any,
        confidence_threshold: float,
    ) -> str | None:
        return HuggingFaceNMTModel._extract_suggestion_text(
            self,
            tokenizer,
            partial_translation,
            translation_token_ids,
            token_scores,
            confidence_threshold,
        )


class SuggestionService:
    """Produce an inline continuation of a partial translation."""

    def __init__(self, model: NllbModel) -> None:
        self._model = model
        self._suggesters: Dict[Tuple[str, str], TranslationSuggester] = {}

    def _get_suggester(self, src_lang: str, tgt_lang: str) -> TranslationSuggester:
        key = (src_lang, tgt_lang)
        if key in self._suggesters:
            return self._suggesters[key]

        pipeline_device = 0 if self._model.device.type == "cuda" else -1
        pipeline = SilTranslationPipeline(
            model=self._model.model,
            tokenizer=self._model.tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=pipeline_device,
        )
        suggester = HuggingFaceTranslationSuggester(
            _NllbSuggesterModelAdapter(),
            pipeline,
            self._model.tokenizer,
            confidence_threshold=0.25,
            max_new_tokens=64,
            num_beams=1,
            constraint_indexes=self._model.constraint_indexes,
        )
        self._suggesters[key] = suggester
        return suggester

    def suggest(
        self,
        source_text: str,
        partial_translation: str,
        src_lang: str,
        tgt_lang: str,
        confidence_threshold: float,
    ) -> str:
        if not source_text.strip():
            return ""
        with self._model.lock:
            self._model.tokenizer.src_lang = src_lang
            self._model.tokenizer.tgt_lang = tgt_lang
            suggester = self._get_suggester(src_lang, tgt_lang)
            suggester._confidence_threshold = confidence_threshold
            suggestion = suggester.suggestion_translation(source_text, partial_translation)
        if suggestion is None:
            return ""
        return _ensure_utf8_text(suggestion)
