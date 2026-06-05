import logging
from dataclasses import dataclass, field
from math import exp
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

LOGGER = logging.getLogger(__name__)

DEFAULT_LOW_PROB_THRESHOLD = -3.0
DEFAULT_TOP_K_SUGGESTIONS = 5
DEFAULT_MAX_PHRASE_WORDS = 3
DEFAULT_MIN_SUGGESTION_IMPROVEMENT = 0.25
DEFAULT_GENERATION_BEAMS = 8
DEFAULT_MAX_CANDIDATE_TOKENS = 12


@dataclass
class TokenScore:
    """Score information for a single subword token."""

    token: str
    log_prob: float

    @property
    def prob(self) -> float:
        return exp(self.log_prob)


@dataclass
class PhraseSuggestion:
    """A replacement phrase suggestion scored with both left and right context."""

    phrase: str
    forward_log_prob: float
    right_context_log_prob: float
    contextual_log_prob: float
    improvement: float

    @property
    def word_count(self) -> int:
        return max(len(self.phrase.split()), 1)

    @property
    def normalized_log_prob(self) -> float:
        return self.contextual_log_prob / self.word_count


@dataclass
class WordScore:
    """Context-aware score information for a single word."""

    word: str
    tokens: List[TokenScore]
    forward_log_prob: float
    right_context_log_prob: float
    contextual_log_prob: float
    span_start: int
    span_end: int
    suggestions: List[PhraseSuggestion] = field(default_factory=list)
    low_prob_threshold: float = field(default=DEFAULT_LOW_PROB_THRESHOLD, compare=False, repr=False)

    @property
    def word_count(self) -> int:
        return 1

    @property
    def log_prob(self) -> float:
        return self.contextual_log_prob

    @property
    def normalized_log_prob(self) -> float:
        return self.contextual_log_prob

    @property
    def prob(self) -> float:
        return exp(self.contextual_log_prob)

    @property
    def is_low_probability(self) -> bool:
        return self.normalized_log_prob < self.low_prob_threshold


@dataclass
class PhraseScore:
    """Context-aware score for a variable-length phrase span."""

    phrase: str
    word_scores: List[WordScore]
    forward_log_prob: float
    right_context_log_prob: float
    contextual_log_prob: float
    suggestions: List[PhraseSuggestion] = field(default_factory=list)
    low_prob_threshold: float = field(default=DEFAULT_LOW_PROB_THRESHOLD, compare=False, repr=False)

    @property
    def span_start(self) -> int:
        return self.word_scores[0].span_start

    @property
    def span_end(self) -> int:
        return self.word_scores[-1].span_end

    @property
    def word_count(self) -> int:
        return len(self.word_scores)

    @property
    def normalized_log_prob(self) -> float:
        return self.contextual_log_prob / max(self.word_count, 1)

    @property
    def is_low_probability(self) -> bool:
        return self.normalized_log_prob < self.low_prob_threshold


@dataclass
class ScoredTranslation:
    """Result of scoring a translation against a source sentence."""

    source: str
    translation: str
    word_scores: List[WordScore]
    phrase_scores: List[PhraseScore]

    @property
    def sequence_log_prob(self) -> float:
        return sum(w.forward_log_prob for w in self.word_scores)

    @property
    def low_probability_words(self) -> List[WordScore]:
        return [w for w in self.word_scores if w.is_low_probability]

    @property
    def low_probability_phrases(self) -> List[PhraseScore]:
        return [p for p in self.phrase_scores if p.is_low_probability]


@dataclass
class WordSpan:
    text: str
    token_start: int
    token_end: int


@dataclass
class SequenceScore:
    text: str
    label_ids: List[int]
    tokens: List[str]
    token_log_probs: List[float]
    words: List[WordSpan]

    def __post_init__(self) -> None:
        self._word_forward_log_probs = [
            sum(self.token_log_probs[word.token_start : word.token_end]) for word in self.words
        ]
        cumulative = [0.0]
        total = 0.0
        for value in self._word_forward_log_probs:
            total += value
            cumulative.append(total)
        self._cumulative_word_forward_log_probs = cumulative

    @property
    def word_texts(self) -> List[str]:
        return [word.text for word in self.words]

    def phrase_text(self, start: int, end: int) -> str:
        return " ".join(self.word_texts[start:end])

    def word_forward_log_prob(self, start: int, end: int) -> float:
        return self._cumulative_word_forward_log_probs[end] - self._cumulative_word_forward_log_probs[start]


class TranslationScorer:
    """Scores translations with contextual phrase rescoring and replacement suggestions.

    The left-to-right decoder supplies token log probabilities for the observed phrase.
    To incorporate right-context evidence, each phrase is rescored against the observed
    suffix using a contextual span objective:

        contextual(span) = log P(span | prefix, x)
                         + log P(suffix | prefix + span, x)
                         - log P(suffix | prefix, x)

    This acts like a pointwise contextual compatibility score. The same score, normalized
    by phrase length in words, is used for both anomaly detection and for ranking
    replacement candidates of different lengths.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        low_prob_threshold: float = DEFAULT_LOW_PROB_THRESHOLD,
        top_k_suggestions: int = DEFAULT_TOP_K_SUGGESTIONS,
        max_phrase_words: int = DEFAULT_MAX_PHRASE_WORDS,
        min_suggestion_improvement: float = DEFAULT_MIN_SUGGESTION_IMPROVEMENT,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._low_prob_threshold = low_prob_threshold
        self._top_k_suggestions = top_k_suggestions
        self._max_phrase_words = max_phrase_words
        self._min_suggestion_improvement = min_suggestion_improvement
        self._special_token_ids: Set[int] = set(tokenizer.all_special_ids)

    def score(self, source: str, translation: str) -> ScoredTranslation:
        source_context = self._encode_source(source)
        cache: Dict[str, SequenceScore] = {}
        base = self._score_translation_text(source_context, translation, cache)

        phrase_scores = self._build_phrase_scores(source_context, base, cache)
        word_scores = [
            self._to_word_score(phrase_score) for phrase_score in phrase_scores if phrase_score.word_count == 1
        ]
        multi_word_phrase_scores = [phrase_score for phrase_score in phrase_scores if phrase_score.word_count > 1]

        return ScoredTranslation(
            source=source,
            translation=translation,
            word_scores=word_scores,
            phrase_scores=multi_word_phrase_scores,
        )

    def _build_phrase_scores(
        self,
        source_context: Dict[str, torch.Tensor],
        base: SequenceScore,
        cache: Dict[str, SequenceScore],
    ) -> List[PhraseScore]:
        phrase_scores: List[PhraseScore] = []

        for phrase_len in range(1, min(self._max_phrase_words, len(base.words)) + 1):
            for start in range(0, len(base.words) - phrase_len + 1):
                end = start + phrase_len
                phrase_text = base.phrase_text(start, end)
                forward_log_prob = base.word_forward_log_prob(start, end)
                right_context_log_prob, ablated = self._score_right_context(
                    source_context,
                    base,
                    start,
                    end,
                    cache,
                )
                contextual_log_prob = forward_log_prob + right_context_log_prob
                span_word_scores = self._create_span_word_scores(base, start, end, right_context_log_prob)
                phrase_score = PhraseScore(
                    phrase=phrase_text,
                    word_scores=span_word_scores,
                    forward_log_prob=forward_log_prob,
                    right_context_log_prob=right_context_log_prob,
                    contextual_log_prob=contextual_log_prob,
                    low_prob_threshold=self._low_prob_threshold,
                )
                if phrase_score.is_low_probability:
                    phrase_score.suggestions = self._generate_scored_suggestions(
                        source_context,
                        base,
                        ablated,
                        phrase_score,
                        cache,
                    )
                phrase_scores.append(phrase_score)
        # Reuse phrase-level suggestions for the singleton word scores.
        for phrase_score in phrase_scores:
            if phrase_score.word_count == 1:
                continue
            for word_score in phrase_score.word_scores:
                if not word_score.suggestions and phrase_score.suggestions:
                    word_score.suggestions = phrase_score.suggestions

        return phrase_scores

    def _create_span_word_scores(
        self, base: SequenceScore, start: int, end: int, right_context_log_prob: float
    ) -> List[WordScore]:
        span = base.words[start:end]
        span_forward = base.word_forward_log_prob(start, end)
        span_contextual = span_forward + right_context_log_prob
        word_count = max(end - start, 1)
        span_share = right_context_log_prob / word_count
        word_scores: List[WordScore] = []
        for index, word in enumerate(span, start):
            tokens = [
                TokenScore(token=base.tokens[token_index], log_prob=base.token_log_probs[token_index])
                for token_index in range(word.token_start, word.token_end)
            ]
            word_forward = base.word_forward_log_prob(index, index + 1)
            word_scores.append(
                WordScore(
                    word=word.text,
                    tokens=tokens,
                    forward_log_prob=word_forward,
                    right_context_log_prob=span_share,
                    contextual_log_prob=(
                        span_contextual / word_count if word_count > 1 else word_forward + right_context_log_prob
                    ),
                    span_start=index,
                    span_end=index + 1,
                    low_prob_threshold=self._low_prob_threshold,
                )
            )
        return word_scores

    def _to_word_score(self, phrase_score: PhraseScore) -> WordScore:
        word_score = phrase_score.word_scores[0]
        word_score.suggestions = phrase_score.suggestions
        return word_score

    def _score_right_context(
        self,
        source_context: Dict[str, torch.Tensor],
        base: SequenceScore,
        start: int,
        end: int,
        cache: Dict[str, SequenceScore],
    ) -> Tuple[float, SequenceScore]:
        suffix_with_phrase = base.word_forward_log_prob(end, len(base.words))
        ablated_words = base.word_texts[:start] + base.word_texts[end:]
        ablated_text = self._join_words(ablated_words)
        ablated = self._score_translation_text(source_context, ablated_text, cache)
        if end >= len(base.words):
            return 0.0, ablated
        suffix_without_phrase = ablated.word_forward_log_prob(start, len(ablated.words))
        return suffix_with_phrase - suffix_without_phrase, ablated

    def _generate_scored_suggestions(
        self,
        source_context: Dict[str, torch.Tensor],
        base: SequenceScore,
        ablated: SequenceScore,
        phrase_score: PhraseScore,
        cache: Dict[str, SequenceScore],
    ) -> List[PhraseSuggestion]:
        prefix_words = base.word_texts[: phrase_score.span_start]
        suffix_words = base.word_texts[phrase_score.span_end :]
        prefix_text = self._join_words(prefix_words)
        candidate_phrases = self._generate_candidate_phrases(source_context, prefix_text)
        suggestions: List[PhraseSuggestion] = []
        seen: Set[str] = set()

        for candidate_phrase in candidate_phrases:
            normalized_candidate = candidate_phrase.strip()
            if (
                normalized_candidate == ""
                or normalized_candidate == phrase_score.phrase
                or normalized_candidate in seen
            ):
                continue
            seen.add(normalized_candidate)
            candidate_words = normalized_candidate.split()
            variant_words = prefix_words + candidate_words + suffix_words
            variant_text = self._join_words(variant_words)
            variant = self._score_translation_text(source_context, variant_text, cache)
            candidate_end = phrase_score.span_start + len(candidate_words)
            forward_log_prob = variant.word_forward_log_prob(phrase_score.span_start, candidate_end)
            suffix_with_candidate = variant.word_forward_log_prob(candidate_end, len(variant.words))
            suffix_without_phrase = ablated.word_forward_log_prob(phrase_score.span_start, len(ablated.words))
            right_context_log_prob = suffix_with_candidate - suffix_without_phrase
            contextual_log_prob = forward_log_prob + right_context_log_prob
            suggestion = PhraseSuggestion(
                phrase=normalized_candidate,
                forward_log_prob=forward_log_prob,
                right_context_log_prob=right_context_log_prob,
                contextual_log_prob=contextual_log_prob,
                improvement=(contextual_log_prob / max(len(candidate_words), 1)) - phrase_score.normalized_log_prob,
            )
            if suggestion.improvement >= self._min_suggestion_improvement:
                suggestions.append(suggestion)

        suggestions.sort(key=lambda suggestion: (suggestion.normalized_log_prob, suggestion.improvement), reverse=True)
        return suggestions[: self._top_k_suggestions]

    def _generate_candidate_phrases(self, source_context: Dict[str, torch.Tensor], prefix_text: str) -> List[str]:
        self._model.eval()
        prefix_ids = self._build_decoder_prefix_ids(prefix_text).to(source_context["input_ids"].device)
        num_beams = max(DEFAULT_GENERATION_BEAMS, self._top_k_suggestions * 2)
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=source_context["input_ids"],
                attention_mask=source_context["attention_mask"],
                decoder_input_ids=prefix_ids,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=DEFAULT_MAX_CANDIDATE_TOKENS,
                early_stopping=True,
                return_dict_in_generate=True,
                forced_bos_token_id=None,
            )

        candidates: List[str] = []
        prefix_len = prefix_ids.shape[1]
        for sequence in outputs.sequences:
            continuation_ids = sequence[prefix_len:]
            candidate_text = self._tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            if candidate_text == "":
                continue
            words = candidate_text.split()
            max_words = min(self._max_phrase_words, len(words))
            for phrase_len in range(1, max_words + 1):
                candidates.append(" ".join(words[:phrase_len]))
        return candidates

    def _encode_source(self, source: str) -> Dict[str, torch.Tensor]:
        source_encoding = self._tokenizer(source, return_tensors="pt", truncation=True)
        device = next(self._model.parameters()).device
        return {
            "input_ids": source_encoding["input_ids"].to(device),
            "attention_mask": source_encoding["attention_mask"].to(device),
        }

    def _score_translation_text(
        self,
        source_context: Dict[str, torch.Tensor],
        translation: str,
        cache: Dict[str, SequenceScore],
    ) -> SequenceScore:
        cached = cache.get(translation)
        if cached is not None:
            return cached

        labels = self._prepare_target_labels(translation)
        labels_on_device = labels.to(source_context["input_ids"].device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(
                input_ids=source_context["input_ids"],
                attention_mask=source_context["attention_mask"],
                labels=labels_on_device,
            )

        logits = outputs.logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        label_ids = labels[0].tolist()
        tokens = [self._tokenizer.convert_ids_to_tokens(token_id) for token_id in label_ids]
        token_log_probs = [log_probs[0, index, token_id].item() for index, token_id in enumerate(label_ids)]
        words = self._group_tokens_into_words(tokens, token_log_probs, label_ids)
        sequence = SequenceScore(
            text=translation,
            label_ids=label_ids,
            tokens=tokens,
            token_log_probs=token_log_probs,
            words=words,
        )
        cache[translation] = sequence
        return sequence

    def _prepare_target_labels(self, translation: str) -> torch.Tensor:
        target_encoding = self._tokenizer(text_target=translation, return_tensors="pt", truncation=True)
        labels = target_encoding["input_ids"]
        # forced_bos_token_id = self._get_forced_bos_token_id()
        forced_bos_token_id = 2
        if forced_bos_token_id is not None and (labels.shape[1] == 0 or labels[0, 0].item() != forced_bos_token_id):
            forced_bos = torch.tensor([[forced_bos_token_id]], dtype=labels.dtype)
            labels = torch.cat([forced_bos, labels], dim=1)
        return labels

    def _build_decoder_prefix_ids(self, prefix_text: str) -> torch.Tensor:
        if prefix_text.strip() == "":
            return torch.tensor([[self._get_initial_decoder_token_id()]], dtype=torch.long)
        prefix_labels = self._prepare_target_labels(prefix_text)
        if prefix_labels.shape[1] > 0 and prefix_labels[0, -1].item() == self._tokenizer.eos_token_id:
            prefix_labels = prefix_labels[:, :-1]
        return prefix_labels

    def _get_initial_decoder_token_id(self) -> int:
        forced_bos_token_id = self._get_forced_bos_token_id()
        if forced_bos_token_id is not None:
            return forced_bos_token_id
        decoder_start_token_id = getattr(self._model.config, "decoder_start_token_id", None)
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        bos_token_id = getattr(self._tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            return bos_token_id
        raise RuntimeError("Unable to determine the initial decoder token id for suggestion generation.")

    def _get_forced_bos_token_id(self) -> Optional[int]:
        if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
            forced_bos_token_id = getattr(self._model.generation_config, "forced_bos_token_id", None)
            if forced_bos_token_id is not None:
                return forced_bos_token_id
        return getattr(self._model.config, "forced_bos_token_id", None)

    def _group_tokens_into_words(
        self,
        tokens: List[str],
        token_log_probs: List[float],
        label_ids: List[int],
    ) -> List[WordSpan]:
        words: List[WordSpan] = []
        current_word_chars = ""
        current_token_start: Optional[int] = None
        current_token_end: Optional[int] = None

        for token_index, (token, _log_prob, label_id) in enumerate(zip(tokens, token_log_probs, label_ids)):
            if label_id in self._special_token_ids:
                if current_token_start is not None and current_token_end is not None:
                    words.append(WordSpan(current_word_chars, current_token_start, current_token_end))
                    current_word_chars = ""
                    current_token_start = None
                    current_token_end = None
                continue

            starts_new_word = self._is_word_initial(token)
            if starts_new_word and current_token_start is not None and current_token_end is not None:
                words.append(WordSpan(current_word_chars, current_token_start, current_token_end))
                current_word_chars = ""
                current_token_start = None
                current_token_end = None

            if current_token_start is None:
                current_token_start = token_index
            current_token_end = token_index + 1
            current_word_chars += self._token_to_word_piece(token)

        if current_token_start is not None and current_token_end is not None:
            words.append(WordSpan(current_word_chars, current_token_start, current_token_end))

        return words

    def _is_word_initial(self, token: str) -> bool:
        return token.startswith("▁") or token.startswith("Ġ")

    def _token_to_word_piece(self, token: str) -> str:
        if token.startswith("▁"):
            return token.removeprefix("▁")
        if token.startswith("Ġ"):
            return token.removeprefix("Ġ")
        if token.startswith("##"):
            return token.removeprefix("##")
        return token

    def _join_words(self, words: List[str]) -> str:
        return " ".join(word for word in words if word != "")
