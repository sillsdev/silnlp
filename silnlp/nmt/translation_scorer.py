"""Contextual scoring of machine translations with replacement suggestions.

This module evaluates how plausible a candidate translation is, token by token,
under a sequence-to-sequence model, flags spans the model finds surprising, and
proposes higher-probability replacements for them.

Design overview
----------------
The pipeline is decomposed into small, single-responsibility collaborators so
that each numerical quantity has exactly one definition:

``ForcedDecoder``
    Runs teacher-forced decoding of a target string against a fixed source and
    returns per-token log probabilities grouped into surface words
    (:class:`TargetSequenceProbabilities`). Also generates candidate
    continuations for a given decoder prefix.

``SpanScorer``
    Turns a base :class:`TargetSequenceProbabilities` into a :class:`SpanScore`
    for any word span.

``AnomalyDetector``
    A strategy that decides which spans are "surprising". Two implementations
    are provided: an absolute per-token threshold and a sentence-relative
    z-score detector (the library default).

``CandidateGenerator`` / ``SuggestionScorer``
    Propose replacement phrases for a flagged span and rank them by how much
    they improve the model's per-token assessment of the whole translation.

Scoring conventions
--------------------
The primary signal is the **mean per-token log probability** of a span,
``log P(span | left context, source) / number_of_subword_tokens(span)``. Using
per-token (not per-word) normalization makes scores comparable across spans of
differing surface length and subword fragmentation, and using only the forward
(left-to-right) probability keeps the score independent of the span's position
in the sentence.

``right_context_log_prob`` is reported as a *diagnostic only*: it is the mean
per-token log probability of the next few words and is never used to decide
whether a span is flagged. It is therefore a plain log probability of the
right-context tokens, with no ablation involved.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from math import exp
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput

LOGGER = logging.getLogger(__name__)

# Anomaly detection.
DEFAULT_LOW_PROB_THRESHOLD = -3.0  # absolute mean-per-token log-prob threshold
DEFAULT_Z_THRESHOLD = 1.5  # flag spans this many std-devs below the sentence mean

# Suggestion generation / ranking.
DEFAULT_TOP_K_SUGGESTIONS = 5
DEFAULT_MAX_PHRASE_WORDS = 4
DEFAULT_MIN_SUGGESTION_IMPROVEMENT = 0.05  # minimum nats-per-token improvement to keep a suggestion
# Standard beam search (num_beam_groups=1) processes all beams in a single GPU
# forward pass per step. Diverse beam search processes each group sequentially,
# multiplying the number of CPU-GPU round-trips by the group count and leaving
# the GPU idle between groups — this was the primary source of latency.
DEFAULT_GENERATION_BEAMS = 4
DEFAULT_BEAM_GROUPS = 1
DEFAULT_DIVERSITY_PENALTY = 0.0
DEFAULT_MAX_CANDIDATE_TOKENS = 8

# Right-context diagnostic.
DEFAULT_RIGHT_CONTEXT_WINDOW = 5  # number of following words summarized by the diagnostic


# ---------------------------------------------------------------------------
# Teacher-forced decoding: per-token probabilities grouped into words.
# ---------------------------------------------------------------------------


@dataclass
class TokenProbability:
    """Log probability the model assigned to a single observed subword token."""

    token: str
    log_prob: float

    @property
    def prob(self) -> float:
        return exp(self.log_prob)


@dataclass
class WordTokenSpan:
    """A surface word and the half-open range of subword tokens that compose it."""

    text: str
    token_start: int
    token_end: int

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass
class TargetSequenceProbabilities:
    """Per-token log probabilities for one teacher-forced target sequence.

    ``words`` references only content tokens; structural tokens (language tag,
    EOS, padding) are excluded so that every reported quantity reflects the
    translation's actual words.
    """

    text: str
    tokens: List[str]
    token_log_probs: List[float]
    words: List[WordTokenSpan]

    def __post_init__(self) -> None:
        cumulative_log_prob = [0.0]
        cumulative_token_count = [0]
        for word in self.words:
            span_log_prob = sum(self.token_log_probs[word.token_start : word.token_end])
            cumulative_log_prob.append(cumulative_log_prob[-1] + span_log_prob)
            cumulative_token_count.append(cumulative_token_count[-1] + word.token_count)
        self._cumulative_log_prob = cumulative_log_prob
        self._cumulative_token_count = cumulative_token_count

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def word_texts(self) -> List[str]:
        return [word.text for word in self.words]

    def phrase_text(self, word_start: int, word_end: int) -> str:
        return " ".join(self.word_texts[word_start:word_end])

    def log_prob(self, word_start: int, word_end: int) -> float:
        """Summed log probability of the words in ``[word_start, word_end)``."""
        return self._cumulative_log_prob[word_end] - self._cumulative_log_prob[word_start]

    def token_count(self, word_start: int, word_end: int) -> int:
        """Number of subword tokens spanned by the words in ``[word_start, word_end)``."""
        return self._cumulative_token_count[word_end] - self._cumulative_token_count[word_start]

    def token_scores(self, word_start: int, word_end: int) -> List[TokenProbability]:
        scores: List[TokenProbability] = []
        for word in self.words[word_start:word_end]:
            for index in range(word.token_start, word.token_end):
                scores.append(TokenProbability(token=self.tokens[index], log_prob=self.token_log_probs[index]))
        return scores

    @property
    def total_log_prob(self) -> float:
        return self._cumulative_log_prob[-1]

    @property
    def total_token_count(self) -> int:
        return self._cumulative_token_count[-1]

    @property
    def mean_token_log_prob(self) -> float:
        if self.total_token_count == 0:
            return 0.0
        return self.total_log_prob / self.total_token_count


class _SubwordWordGrouper:
    """Groups subword tokens into surface words for a SentencePiece/BPE vocabulary."""

    def __init__(self, structural_token_ids: Set[int]):
        self._structural_token_ids = structural_token_ids

    def group(self, tokens: List[str], label_ids: List[int]) -> List[WordTokenSpan]:
        words: List[WordTokenSpan] = []
        current_text = ""
        current_start: Optional[int] = None
        current_end: Optional[int] = None

        def flush() -> None:
            nonlocal current_text, current_start, current_end
            if current_start is not None and current_end is not None:
                words.append(WordTokenSpan(current_text, current_start, current_end))
            current_text = ""
            current_start = None
            current_end = None

        for index, (token, label_id) in enumerate(zip(tokens, label_ids)):
            if label_id in self._structural_token_ids:
                flush()
                continue
            if self._is_word_initial(token) and current_start is not None:
                flush()
            if current_start is None:
                current_start = index
            current_end = index + 1
            current_text += self._strip_word_marker(token)

        flush()
        return words

    @staticmethod
    def _is_word_initial(token: str) -> bool:
        return token.startswith("▁") or token.startswith("Ġ")

    @staticmethod
    def _strip_word_marker(token: str) -> str:
        for marker in ("▁", "Ġ", "##"):
            if token.startswith(marker):
                return token.removeprefix(marker)
        return token


class ForcedDecoder:
    """Teacher-forced decoding and candidate generation against a fixed source.

    A single source sentence is encoded once via :meth:`set_source`; subsequent
    :meth:`score_target` calls reuse that encoding and are memoized, so repeatedly
    rescoring near-identical variants of a translation is cheap.
    """

    # Maximum number of target sequences scored in one batched forward pass.
    # Larger values reduce latency via parallelism; smaller values reduce peak
    # memory usage (logits are [batch, tgt_len, vocab_size] and can be large).
    SCORING_BATCH_SIZE = 16

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._device = next(model.parameters()).device
        self._forced_bos_token_id = self._resolve_forced_bos_token_id()
        structural_token_ids: Set[int] = set(tokenizer.all_special_ids)
        if self._forced_bos_token_id is not None:
            structural_token_ids.add(self._forced_bos_token_id)
        self._grouper = _SubwordWordGrouper(structural_token_ids)
        self._encoder_outputs: Optional[BaseModelOutput] = None
        self._source_attention_mask: Optional[torch.Tensor] = None
        self._cache: Dict[str, TargetSequenceProbabilities] = {}

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def _require_source(self) -> Tuple[BaseModelOutput, torch.Tensor]:
        if self._encoder_outputs is None or self._source_attention_mask is None:
            raise RuntimeError("set_source() must be called before decoding.")
        return self._encoder_outputs, self._source_attention_mask

    def set_source(self, source: str) -> None:
        encoding = self._tokenizer(source, return_tensors="pt", truncation=True)
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)
        # Run the encoder exactly once and cache its output.
        self._model.eval()
        with torch.no_grad():
            self._encoder_outputs = self._model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        self._source_attention_mask = attention_mask
        self._cache = {}

    def score_target(self, target_text: str) -> TargetSequenceProbabilities:
        cached = self._cache.get(target_text)
        if cached is not None:
            return cached
        return self.score_targets_batch([target_text])[0]

    def score_targets_batch(self, target_texts: List[str]) -> List[TargetSequenceProbabilities]:
        """Score multiple target sequences in as few forward passes as possible.

        Sequences already in the cache are returned immediately. Uncached
        sequences are scored in chunks of :attr:`SCORING_BATCH_SIZE` using a
        single padded batched forward pass per chunk, reusing the cached encoder
        output so the encoder runs zero additional times.
        """
        encoder_outputs, attention_mask = self._require_source()

        results: List[Optional[TargetSequenceProbabilities]] = [None] * len(target_texts)
        uncached_indices: List[int] = []
        for i, text in enumerate(target_texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)

        # Process uncached texts in chunks to bound peak memory.
        for chunk_start in range(0, len(uncached_indices), self.SCORING_BATCH_SIZE):
            chunk_indices = uncached_indices[chunk_start : chunk_start + self.SCORING_BATCH_SIZE]
            chunk_texts = [target_texts[i] for i in chunk_indices]
            chunk_labels = [self._target_labels(t) for t in chunk_texts]

            max_len = max(lbl.shape[1] for lbl in chunk_labels)
            pad_id = self._tokenizer.eos_token_id or 0
            padded = []
            for lbl in chunk_labels:
                pad_len = max_len - lbl.shape[1]
                if pad_len:
                    lbl = torch.cat([lbl, lbl.new_full((1, pad_len), pad_id)], dim=1)
                padded.append(lbl)
            batch_labels = torch.cat(padded, dim=0).to(self._device)

            batch_size = len(chunk_texts)
            batch_encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state.expand(batch_size, -1, -1),
            )
            batch_attention_mask = attention_mask.expand(batch_size, -1)

            self._model.eval()
            with torch.no_grad():
                outputs = self._model(
                    encoder_outputs=batch_encoder_outputs,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                )

            # log_probs shape: [batch, max_len, vocab]; discard immediately after use.
            log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

            for j, (text, orig_labels) in enumerate(zip(chunk_texts, chunk_labels)):
                seq_len = orig_labels.shape[1]
                label_ids = orig_labels[0].tolist()
                tokens = [self._tokenizer.convert_ids_to_tokens(tid) for tid in label_ids]
                token_log_probs = [log_probs[j, pos, tid].item() for pos, tid in enumerate(label_ids)]
                words = self._grouper.group(tokens, label_ids)
                sequence = TargetSequenceProbabilities(
                    text=text,
                    tokens=tokens,
                    token_log_probs=token_log_probs,
                    words=words,
                )
                self._cache[text] = sequence
                results[chunk_indices[j]] = sequence

            del log_probs  # free memory before next chunk

        return results  # type: ignore[return-value]  # all slots filled above

    def generate_continuations(
        self,
        prefix_text: str,
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        max_new_tokens: int,
    ) -> List[str]:
        prefix_ids = self._decoder_prefix_ids(prefix_text).to(self._device)
        results = self._generate_from_prefix_batch(
            prefix_ids_batch=prefix_ids,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            max_new_tokens=max_new_tokens,
        )
        return results[0]

    def generate_continuations_for_prefixes(
        self,
        prefix_texts: List[str],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        max_new_tokens: int,
    ) -> List[List[str]]:
        """Generate continuations for multiple decoder prefixes in as few
        ``model.generate`` calls as possible.

        Prefixes of identical tokenized length are batched together into a single
        generate call, avoiding both encoder re-runs (encoder is cached) and
        repeated decoder-setup overhead. Prefixes of different lengths are grouped
        separately so no padding is needed.
        """
        # Tokenize all prefixes and group by token length.
        prefix_ids_list = [self._decoder_prefix_ids(t).to(self._device) for t in prefix_texts]
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, ids in enumerate(prefix_ids_list):
            groups[ids.shape[1]].append(i)

        results: List[Optional[List[str]]] = [None] * len(prefix_texts)
        for length, indices in groups.items():
            batch = torch.cat([prefix_ids_list[i] for i in indices], dim=0)
            batch_results = self._generate_from_prefix_batch(
                prefix_ids_batch=batch,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                max_new_tokens=max_new_tokens,
            )
            for i, continuations in zip(indices, batch_results):
                results[i] = continuations

        return results  # type: ignore[return-value]

    def _generate_from_prefix_batch(
        self,
        prefix_ids_batch: torch.Tensor,
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        max_new_tokens: int,
    ) -> List[List[str]]:
        """Run one batched generate call; return continuations per prefix."""
        encoder_outputs, attention_mask = self._require_source()
        batch_size = prefix_ids_batch.shape[0]

        # Expand to batch_size only — model.generate handles the further
        # expansion by num_beams internally during beam search setup.
        batch_encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state.expand(batch_size, -1, -1),
        )
        batch_attention_mask = attention_mask.expand(batch_size, -1)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model.generate(
                encoder_outputs=batch_encoder_outputs,
                attention_mask=batch_attention_mask,
                decoder_input_ids=prefix_ids_batch,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                num_return_sequences=num_beams,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                forced_bos_token_id=None,
            )

        # model.generate prepends decoder_start_token_id, so the continuation
        # starts at position 1 + prefix_length in each output sequence.
        prefix_length = 1 + prefix_ids_batch.shape[1]
        # outputs.sequences shape: [batch_size * num_beams, total_length]
        all_continuations: List[List[str]] = [[] for _ in range(batch_size)]
        for seq_idx, sequence in enumerate(outputs.sequences):
            prefix_idx = seq_idx // num_beams
            continuation_ids = sequence[prefix_length:]
            text = self._tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            if text:
                all_continuations[prefix_idx].append(text)
        return all_continuations

    def _target_labels(self, target_text: str) -> torch.Tensor:
        encoding = self._tokenizer(text_target=target_text, return_tensors="pt", truncation=True)
        labels = encoding["input_ids"]
        forced_bos = self._forced_bos_token_id
        if forced_bos is not None and (labels.shape[1] == 0 or labels[0, 0].item() != forced_bos):
            prefix = torch.tensor([[forced_bos]], dtype=labels.dtype)
            labels = torch.cat([prefix, labels], dim=1)
        return labels

    def _decoder_prefix_ids(self, prefix_text: str) -> torch.Tensor:
        if prefix_text.strip() == "":
            return torch.tensor([[self._initial_decoder_token_id()]], dtype=torch.long)
        prefix_labels = self._target_labels(prefix_text)
        if prefix_labels.shape[1] > 0 and prefix_labels[0, -1].item() == self._tokenizer.eos_token_id:
            prefix_labels = prefix_labels[:, :-1]
        return prefix_labels

    def _initial_decoder_token_id(self) -> int:
        if self._forced_bos_token_id is not None:
            return self._forced_bos_token_id
        decoder_start_token_id = getattr(self._model.config, "decoder_start_token_id", None)
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        bos_token_id = getattr(self._tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            return bos_token_id
        raise RuntimeError("Unable to determine the initial decoder token id for suggestion generation.")

    def _resolve_forced_bos_token_id(self) -> Optional[int]:
        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is not None:
            forced_bos = getattr(generation_config, "forced_bos_token_id", None)
            if forced_bos is not None:
                return forced_bos
        return getattr(self._model.config, "forced_bos_token_id", None)


# ---------------------------------------------------------------------------
# Span scoring.
# ---------------------------------------------------------------------------


@dataclass
class ReplacementSuggestion:
    """A proposed replacement phrase, ranked by per-token improvement."""

    phrase: str
    mean_token_log_prob: float  # variant's whole-sentence mean per-token log prob
    improvement: float  # nats per token gained over the original translation


@dataclass
class SpanScore:
    """Score for a contiguous span of words in the translation.

    The flagging decision uses only :attr:`mean_token_log_prob`. ``forward_log_prob``
    is its un-normalized counterpart, and ``right_context_log_prob`` is a reported
    diagnostic (see the module docstring), never an input to flagging.
    """

    text: str
    word_start: int
    word_end: int
    token_count: int
    forward_log_prob: float
    mean_token_log_prob: float
    right_context_log_prob: float
    tokens: List[TokenProbability] = field(default_factory=list)
    is_flagged: bool = False
    suggestions: List[ReplacementSuggestion] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return self.word_end - self.word_start


class SpanScorer:
    """Builds :class:`SpanScore` objects from a base sequence's probabilities."""

    def __init__(self, right_context_window: int = DEFAULT_RIGHT_CONTEXT_WINDOW):
        self._right_context_window = right_context_window

    def score(self, base: TargetSequenceProbabilities, word_start: int, word_end: int) -> SpanScore:
        forward_log_prob = base.log_prob(word_start, word_end)
        token_count = base.token_count(word_start, word_end)
        return SpanScore(
            text=base.phrase_text(word_start, word_end),
            word_start=word_start,
            word_end=word_end,
            token_count=token_count,
            forward_log_prob=forward_log_prob,
            mean_token_log_prob=forward_log_prob / token_count if token_count else 0.0,
            right_context_log_prob=self._right_context_mean(base, word_end),
            tokens=base.token_scores(word_start, word_end),
        )

    def _right_context_mean(self, base: TargetSequenceProbabilities, word_end: int) -> float:
        window_end = min(word_end + self._right_context_window, base.word_count)
        window_token_count = base.token_count(word_end, window_end)
        if window_token_count == 0:
            return 0.0
        return base.log_prob(word_end, window_end) / window_token_count


# ---------------------------------------------------------------------------
# Anomaly detection strategies.
# ---------------------------------------------------------------------------


class AnomalyDetector(ABC):
    """Decides which spans are surprising, given the population of word scores.

    :meth:`threshold` is computed once per sentence from the per-token mean log
    probabilities of its words; a span is flagged when its own mean falls below
    the returned threshold.
    """

    @abstractmethod
    def threshold(self, word_mean_log_probs: Sequence[float]) -> float:
        ...


class AbsoluteThresholdAnomalyDetector(AnomalyDetector):
    """Flags spans whose mean per-token log probability is below a fixed value."""

    def __init__(self, threshold: float = DEFAULT_LOW_PROB_THRESHOLD):
        self._threshold = threshold

    def threshold(self, word_mean_log_probs: Sequence[float]) -> float:
        return self._threshold


class ZScoreAnomalyDetector(AnomalyDetector):
    """Flags spans more than ``z`` standard deviations below the sentence mean.

    This is language- and model-agnostic: it adapts to each sentence's own
    distribution of token probabilities rather than relying on a corpus-specific
    absolute cutoff. Sentences too short to estimate a spread flag nothing.
    """

    def __init__(self, z: float = DEFAULT_Z_THRESHOLD):
        self._z = z

    def threshold(self, word_mean_log_probs: Sequence[float]) -> float:
        if len(word_mean_log_probs) < 2:
            return float("-inf")
        spread = pstdev(word_mean_log_probs)
        if spread == 0.0:
            return float("-inf")
        return mean(word_mean_log_probs) - self._z * spread


# ---------------------------------------------------------------------------
# Replacement suggestions.
# ---------------------------------------------------------------------------


class CandidateGenerator:
    """Generates diverse replacement phrases for a flagged span via beam search.

    Beam search is seeded with the decoder prefix (the words before the span) and
    produces, per beam, all word-length prefixes of the continuation up to
    ``max_phrase_words`` words — e.g. a beam returning "issued a decree" yields
    "issued", "issued a", and "issued a decree" as separate candidates. Diverse
    beam groups reduce near-duplicate hypotheses; case-insensitive deduplication
    removes the rest.
    """

    def __init__(
        self,
        decoder: ForcedDecoder,
        max_phrase_words: int = DEFAULT_MAX_PHRASE_WORDS,
        num_beams: int = DEFAULT_GENERATION_BEAMS,
        num_beam_groups: int = DEFAULT_BEAM_GROUPS,
        diversity_penalty: float = DEFAULT_DIVERSITY_PENALTY,
        max_new_tokens: int = DEFAULT_MAX_CANDIDATE_TOKENS,
    ):
        self._decoder = decoder
        self._max_phrase_words = max_phrase_words
        self._num_beam_groups = max(1, min(num_beam_groups, num_beams))
        # Diverse beam search requires the beam count to be a multiple of the group count.
        self._num_beams = max(self._num_beam_groups, (num_beams // self._num_beam_groups) * self._num_beam_groups)
        self._diversity_penalty = diversity_penalty if self._num_beam_groups > 1 else 0.0
        self._max_new_tokens = max_new_tokens

    def generate(self, prefix_text: str) -> List[str]:
        return self.generate_batch([prefix_text])[0]

    def generate_batch(self, prefix_texts: List[str]) -> List[List[str]]:
        """Generate candidates for multiple prefixes in as few generate calls as possible."""
        all_continuations = self._decoder.generate_continuations_for_prefixes(
            prefix_texts,
            num_beams=self._num_beams,
            num_beam_groups=self._num_beam_groups,
            diversity_penalty=self._diversity_penalty,
            max_new_tokens=self._max_new_tokens,
        )
        return [self._deduplicate_candidates(continuations) for continuations in all_continuations]

    def _deduplicate_candidates(self, continuations: List[str]) -> List[str]:
        candidates: List[str] = []
        seen: Set[str] = set()
        for continuation in continuations:
            words = continuation.split()[: self._max_phrase_words]
            for length in range(1, len(words) + 1):
                phrase = " ".join(words[:length])
                key = phrase.casefold()
                if key not in seen:
                    seen.add(key)
                    candidates.append(phrase)
        return candidates


class SuggestionScorer:
    """Scores and ranks replacement candidates for a flagged span.

    Each candidate is spliced into the full translation and the whole variant is
    rescored. The ranking metric is the change in the translation's mean
    per-token log probability, which is directly comparable across candidates of
    any length.
    """

    def __init__(
        self,
        decoder: ForcedDecoder,
        min_improvement: float = DEFAULT_MIN_SUGGESTION_IMPROVEMENT,
        top_k: int = DEFAULT_TOP_K_SUGGESTIONS,
    ):
        self._decoder = decoder
        self._min_improvement = min_improvement
        self._top_k = top_k

    def score(
        self,
        base: TargetSequenceProbabilities,
        span: SpanScore,
        candidate_phrases: Sequence[str],
    ) -> List[ReplacementSuggestion]:
        prefix_words = base.word_texts[: span.word_start]
        suffix_words = base.word_texts[span.word_end :]
        original_mean = base.mean_token_log_prob

        suggestions: List[ReplacementSuggestion] = []
        seen: Set[str] = set()
        for candidate in candidate_phrases:
            phrase = candidate.strip()
            key = phrase.casefold()
            if phrase == "" or key == span.text.casefold() or key in seen:
                continue
            seen.add(key)

            variant_text = self._join(prefix_words + phrase.split() + suffix_words)
            variant = self._decoder.score_target(variant_text)
            improvement = variant.mean_token_log_prob - original_mean
            if improvement >= self._min_improvement:
                suggestions.append(
                    ReplacementSuggestion(
                        phrase=phrase,
                        mean_token_log_prob=variant.mean_token_log_prob,
                        improvement=improvement,
                    )
                )

        suggestions.sort(key=lambda suggestion: suggestion.improvement, reverse=True)
        return suggestions[: self._top_k]

    @staticmethod
    def _join(words: List[str]) -> str:
        return " ".join(word for word in words if word != "")


# ---------------------------------------------------------------------------
# Top-level result and orchestration.
# ---------------------------------------------------------------------------


@dataclass
class ScoredTranslation:
    """The result of scoring a translation against a source sentence."""

    source: str
    translation: str
    word_scores: List[SpanScore]
    phrase_scores: List[SpanScore]

    @property
    def total_log_prob(self) -> float:
        """Log probability of the whole translation (summed over its words)."""
        return sum(score.forward_log_prob for score in self.word_scores)

    @property
    def mean_token_log_prob(self) -> float:
        total_tokens = sum(score.token_count for score in self.word_scores)
        if total_tokens == 0:
            return 0.0
        return self.total_log_prob / total_tokens

    @property
    def flagged_words(self) -> List[SpanScore]:
        return [score for score in self.word_scores if score.is_flagged]

    @property
    def flagged_phrases(self) -> List[SpanScore]:
        return [score for score in self.phrase_scores if score.is_flagged]


class TranslationScorer:
    """Scores a translation and suggests replacements for surprising spans.

    Args:
        model: A seq2seq model whose ``generation_config``/``config`` already
            encodes the target language (e.g. NLLB ``forced_bos_token_id``).
        tokenizer: The matching tokenizer.
        anomaly_detector: Strategy that decides which spans are flagged. Defaults
            to :class:`ZScoreAnomalyDetector`.
        top_k_suggestions: Maximum replacement suggestions per flagged span.
        max_phrase_words: Longest span (in words) considered for flagging and the
            longest replacement phrase generated.
        min_suggestion_improvement: Minimum nats-per-token improvement for a
            suggestion to be kept.
        right_context_window: Number of following words summarized by the
            reported right-context diagnostic.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        anomaly_detector: Optional[AnomalyDetector] = None,
        top_k_suggestions: int = DEFAULT_TOP_K_SUGGESTIONS,
        max_phrase_words: int = DEFAULT_MAX_PHRASE_WORDS,
        min_suggestion_improvement: float = DEFAULT_MIN_SUGGESTION_IMPROVEMENT,
        right_context_window: int = DEFAULT_RIGHT_CONTEXT_WINDOW,
    ):
        self._decoder = ForcedDecoder(model, tokenizer)
        self._detector = anomaly_detector or ZScoreAnomalyDetector()
        self._span_scorer = SpanScorer(right_context_window)
        self._candidate_generator = CandidateGenerator(self._decoder, max_phrase_words=max_phrase_words)
        self._suggestion_scorer = SuggestionScorer(
            self._decoder,
            min_improvement=min_suggestion_improvement,
            top_k=top_k_suggestions,
        )
        self._max_phrase_words = max_phrase_words

    def score(self, source: str, translation: str) -> ScoredTranslation:
        t0 = time.perf_counter()
        self._decoder.set_source(source)
        base = self._decoder.score_target(translation)
        LOGGER.info("Initial encode+score: %.3f s", time.perf_counter() - t0)

        # Strategy 3: score single-word spans first, detect anomalies, then
        # score multi-word spans only where flagged words create them.
        word_scores = [self._span_scorer.score(base, i, i + 1) for i in range(base.word_count)]
        word_means = [s.mean_token_log_prob for s in word_scores]
        threshold = self._detector.threshold(word_means)

        flagged_positions: Set[int] = set()
        for span in word_scores:
            span.is_flagged = span.mean_token_log_prob < threshold
            if span.is_flagged:
                flagged_positions.add(span.word_start)

        phrase_scores: List[SpanScore] = []
        if flagged_positions and self._max_phrase_words > 1:
            max_span = min(self._max_phrase_words, base.word_count)
            for span_len in range(2, max_span + 1):
                for start in range(0, base.word_count - span_len + 1):
                    end = start + span_len
                    # A multi-word phrase can only be flagged if at least one
                    # of its words is flagged (phrase mean is a weighted average
                    # of word means, so it's bounded by the individual mins).
                    if flagged_positions.isdisjoint(range(start, end)):
                        continue
                    phrase = self._span_scorer.score(base, start, end)
                    phrase.is_flagged = phrase.mean_token_log_prob < threshold
                    phrase_scores.append(phrase)

        flagged_word_count = len(flagged_positions)
        LOGGER.info(
            "Spans: %d words total, %d flagged words, %d flagged spans (words+phrases)",
            base.word_count,
            flagged_word_count,
            sum(1 for s in word_scores + phrase_scores if s.is_flagged),
        )
        t0 = time.perf_counter()
        self._attach_suggestions(base, word_scores + phrase_scores)
        LOGGER.info("Suggestion generation: %.3f s", time.perf_counter() - t0)

        return ScoredTranslation(
            source=source,
            translation=translation,
            word_scores=word_scores,
            phrase_scores=phrase_scores,
        )

    def _attach_suggestions(self, base: TargetSequenceProbabilities, spans: List[SpanScore]) -> None:
        """Generate and score replacement candidates for all flagged spans.

        Candidate generation is batched by prefix length (strategy 4) and
        variant scoring is batched across all candidates for all spans (strategy 2).
        """
        flagged = [span for span in spans if span.is_flagged]
        if not flagged:
            return

        # Strategy 4: generate candidates for all flagged spans in batched calls.
        prefix_texts = [" ".join(base.word_texts[: span.word_start]) for span in flagged]
        unique_prefix_lengths = len({tuple(text.split()) for text in prefix_texts})
        t0 = time.perf_counter()
        all_candidates_per_span = self._candidate_generator.generate_batch(prefix_texts)
        LOGGER.info(
            "Candidate generation: %.3f s — %d flagged spans, %d unique prefix lengths",
            time.perf_counter() - t0,
            len(flagged),
            unique_prefix_lengths,
        )

        # Build variant texts for all spans, deduplicating across the whole batch.
        original_mean = base.mean_token_log_prob
        min_improvement = self._suggestion_scorer._min_improvement

        span_data: List[Tuple[List[str], List[str]]] = []  # (phrases, variant_texts) per span
        all_variant_texts: List[str] = []
        variant_text_set: Set[str] = set()

        for span, candidates in zip(flagged, all_candidates_per_span):
            prefix_words = base.word_texts[: span.word_start]
            suffix_words = base.word_texts[span.word_end :]
            seen: Set[str] = {span.text.casefold()}
            phrases: List[str] = []
            variant_texts: List[str] = []
            for phrase in candidates:
                phrase = phrase.strip()
                key = phrase.casefold()
                if not phrase or key in seen:
                    continue
                seen.add(key)
                vtext = SuggestionScorer._join(prefix_words + phrase.split() + suffix_words)
                phrases.append(phrase)
                variant_texts.append(vtext)
                if vtext not in variant_text_set:
                    variant_text_set.add(vtext)
                    all_variant_texts.append(vtext)
            span_data.append((phrases, variant_texts))

        # Strategy 2: score all unique variant texts in one batched forward pass.
        t0 = time.perf_counter()
        scored_variants = self._decoder.score_targets_batch(all_variant_texts)
        LOGGER.info(
            "Variant scoring: %.3f s — %d unique variant texts",
            time.perf_counter() - t0,
            len(all_variant_texts),
        )
        variant_score_map: Dict[str, TargetSequenceProbabilities] = {
            text: score for text, score in zip(all_variant_texts, scored_variants)
        }

        # Distribute results back to each span.
        for span, (phrases, variant_texts) in zip(flagged, span_data):
            suggestions: List[ReplacementSuggestion] = []
            for phrase, vtext in zip(phrases, variant_texts):
                variant = variant_score_map[vtext]
                improvement = variant.mean_token_log_prob - original_mean
                if improvement >= min_improvement:
                    suggestions.append(
                        ReplacementSuggestion(
                            phrase=phrase,
                            mean_token_log_prob=variant.mean_token_log_prob,
                            improvement=improvement,
                        )
                    )
            suggestions.sort(key=lambda s: s.improvement, reverse=True)
            span.suggestions = suggestions[: self._suggestion_scorer._top_k]
