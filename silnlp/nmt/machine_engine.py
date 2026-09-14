"""silnlp's inference adapter over sil-machine's ``HuggingFaceNmtEngine``.

machine's engine loads its own tokenizer from the model path and re-tokenizes any segment it is
handed as a token list. silnlp trains its own SentencePiece vocabulary and feeds the model
pretokenized test files, so both behaviours are replaced here, along with the two places where
machine's own batching would silently drop results. Score alignment and the attention-derived
word alignment are inherited.
"""

import logging
from math import log
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from machine.translation import TranslationResult
from machine.translation.huggingface import HuggingFaceNmtEngine, SilTranslationPipeline
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import TruncationStrategy

from ..common.translation_data_structures import SentenceTranslation, SentenceTranslationGroup

LOGGER = logging.getLogger(__package__ + ".machine_engine")


class PretokenizedTranslationPipeline(SilTranslationPipeline):
    """machine's pipeline, with silnlp's pretokenized inputs and per-call decoding settings."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._generation_overrides: Dict[str, Any] = {}

    def use_generation_settings(self, **overrides: Any) -> None:
        """Decoding settings for subsequent calls, replacing any previously set."""
        self._generation_overrides = overrides

    def preprocess(self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None):
        if all(isinstance(arg, str) for arg in args):
            return super().preprocess(*args, truncation=truncation, src_lang=src_lang, tgt_lang=tgt_lang)

        # Decoding a token list back to text and tokenizing it again, as machine's pipeline does,
        # can change the sequence when the vocabulary carries tokens silnlp added itself.
        batch_tokens = [list(arg) for arg in args]
        inputs = self.tokenizer.pad(
            {"input_ids": [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in batch_tokens]},
            padding=True,
            return_tensors="pt",
        )
        inputs["input_tokens"] = batch_tokens
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        return super()._forward(model_inputs, **{**generate_kwargs, **self._generation_overrides})


class Seq2SeqEngine(HuggingFaceNmtEngine):
    """A machine translation engine over a model and tokenizer that silnlp has already configured.

    machine's builder leaves the sequence confidence at -1 when the decoder reported none, which
    is how a missing score is told apart from a score of zero.
    """

    _UNSET_SEQUENCE_CONFIDENCE = -1.0

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 1,
        oom_batch_size_backoff_mult: float = 1.0,
        **pipeline_kwargs: Any,
    ) -> None:
        # Deliberately not calling super().__init__: it would load a second tokenizer from the model
        # path and reject the language codes that silnlp adds to the tokenizer itself.
        self._pipeline_kwargs = pipeline_kwargs
        self._model = model
        self._model.eval()
        # silnlp caches the inference model between calls, so the engine must not free it.
        self._is_model_owned = False
        self._tokenizer = tokenizer
        # silnlp normalizes punctuation through PunctuationNormalizingTokenizer instead.
        self._mpn = None
        self._batch_size = batch_size
        self._oom_batch_size_backoff_mult = oom_batch_size_backoff_mult
        self._pipeline = self._create_pipeline(self._batch_size)
        self._single_segment_pipeline: Optional[PretokenizedTranslationPipeline] = None

    def _create_pipeline(self, batch_size: int) -> PretokenizedTranslationPipeline:
        return PretokenizedTranslationPipeline(
            model=self._model,
            tokenizer=self._tokenizer,
            mpn=self._mpn,
            batch_size=batch_size,
            **self._pipeline_kwargs,
        )

    def _pipeline_for(self, num_drafts: int) -> PretokenizedTranslationPipeline:
        """machine keeps only the first draft of each segment once its pipeline batches more than
        one segment at a time, so several drafts have to be decoded one segment at a time."""
        if num_drafts <= 1:
            return self._pipeline
        if self._single_segment_pipeline is None:
            LOGGER.info(
                "Decoding %d drafts one segment at a time; batching would discard all but the first.", num_drafts
            )
            self._single_segment_pipeline = self._create_pipeline(1)
        return self._single_segment_pipeline

    def _batch_size_for(self, num_drafts: int) -> int:
        return 1 if num_drafts > 1 else self._batch_size

    def translate_n_batch(
        self, n: int, segments: Sequence[Union[str, Sequence[str]]]
    ) -> Sequence[Sequence[TranslationResult]]:
        """Reimplemented so that recovering from an out-of-memory error keeps this engine's
        pipeline; machine's version rebuilds its own, dropping the pretokenized input path."""
        segments = list(segments)
        while True:
            try:
                results: List[Sequence[TranslationResult]] = []
                for step in range(0, len(segments), self._batch_size):
                    results.extend(self._try_translate_n_batch(n, segments[step : step + self._batch_size]))
                return results
            except torch.cuda.OutOfMemoryError:
                if self._oom_batch_size_backoff_mult >= 0.9999 or self._batch_size <= 1:
                    raise
                self._batch_size = max(int(round(self._batch_size * self._oom_batch_size_backoff_mult)), 1)
                LOGGER.warning("Out of memory; reducing the batch size to %d and retrying.", self._batch_size)
                self._pipeline = self._create_pipeline(self._batch_size)

    def translate_drafts(
        self,
        segments: Sequence[Union[str, Sequence[str]]],
        num_drafts: int = 1,
        **generation_settings: Any,
    ) -> List[SentenceTranslationGroup]:
        """Translate a batch into ``num_drafts`` drafts apiece, under the given decoding settings.

        machine fixes the decoding strategy when the engine is built; silnlp varies it per call so
        that one batch can be decoded by beam search and by sampling.
        """
        pipeline = self._pipeline_for(num_drafts)
        previous_pipeline, self._pipeline = self._pipeline, pipeline
        previous_batch_size, self._batch_size = self._batch_size, self._batch_size_for(num_drafts)
        pipeline.use_generation_settings(**generation_settings)
        try:
            results = self.translate_n_batch(num_drafts, list(segments))
        finally:
            pipeline.use_generation_settings()
            self._pipeline = previous_pipeline
            self._batch_size = previous_batch_size
        return [self.to_draft_group(segment_results) for segment_results in results]

    def to_draft_group(self, results: Sequence[TranslationResult]) -> SentenceTranslationGroup:
        return SentenceTranslationGroup([self._to_sentence_translation(result) for result in results])

    def _to_sentence_translation(self, result: TranslationResult) -> SentenceTranslation:
        return SentenceTranslation(
            result.translation,
            list(result.target_tokens),
            [self._to_log_score(confidence) for confidence in result.confidences],
            self._to_sequence_log_score(result.sequence_confidence),
            # machine strips special tokens, so there is no leading language token to drop.
            starts_with_special_token=False,
        )

    def _to_sequence_log_score(self, sequence_confidence: float) -> Optional[float]:
        if sequence_confidence <= self._UNSET_SEQUENCE_CONFIDENCE or sequence_confidence <= 0:
            return None
        return log(sequence_confidence)

    def _to_log_score(self, confidence: float) -> float:
        # SentenceTranslation holds log probabilities and exponentiates them when writing
        # confidence files; machine has already exponentiated.
        return log(confidence) if confidence > 0 else float("-inf")
