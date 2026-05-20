import logging
from dataclasses import dataclass, field
from math import exp
from typing import List, Optional, Set

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

LOGGER = logging.getLogger(__name__)

DEFAULT_LOW_PROB_THRESHOLD = -3.0
DEFAULT_TOP_K_SUGGESTIONS = 5


@dataclass
class TokenScore:
    """Score information for a single subword token."""

    token: str
    log_prob: float

    @property
    def prob(self) -> float:
        return exp(self.log_prob)


@dataclass
class WordScore:
    """Score information for a single word (may consist of multiple subword tokens)."""

    word: str
    tokens: List[TokenScore]
    suggestions: List[str] = field(default_factory=list)
    low_prob_threshold: float = field(default=DEFAULT_LOW_PROB_THRESHOLD, compare=False, repr=False)

    @property
    def log_prob(self) -> float:
        return sum(t.log_prob for t in self.tokens)

    @property
    def prob(self) -> float:
        return exp(self.log_prob)

    @property
    def is_low_probability(self) -> bool:
        return self.log_prob < self.low_prob_threshold


@dataclass
class ScoredTranslation:
    """Result of scoring a translation against a source sentence."""

    source: str
    translation: str
    word_scores: List[WordScore]

    @property
    def sequence_log_prob(self) -> float:
        return sum(w.log_prob for w in self.word_scores)

    @property
    def low_probability_words(self) -> List[WordScore]:
        return [w for w in self.word_scores if w.is_low_probability]


class TranslationScorer:
    """Scores a translation using forced decoding and identifies low-probability words.

    For each target token y_t in the translation, this class computes the conditional
    probability P(y_t | y_1, ..., y_{t-1}, x) where x is the source sentence. It then
    groups subword tokens into words, flags words that fall below a probability threshold,
    and provides top-k alternative suggestions from the model for each flagged word.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        low_prob_threshold: float = DEFAULT_LOW_PROB_THRESHOLD,
        top_k_suggestions: int = DEFAULT_TOP_K_SUGGESTIONS,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._low_prob_threshold = low_prob_threshold
        self._top_k_suggestions = top_k_suggestions
        self._special_token_ids: Set[int] = set(tokenizer.all_special_ids)

    def score(self, source: str, translation: str) -> ScoredTranslation:
        """Score each token in the translation using forced decoding.

        For each target token y_t, computes P(y_t | y_1, ..., y_{t-1}, x). Low-probability
        words are flagged and paired with top-k alternative suggestions from the model.

        Args:
            source: The source sentence.
            translation: The translation to score.

        Returns:
            A ScoredTranslation with per-word scores and suggestions for low-probability words.
        """
        # Tokenize source
        source_encoding = self._tokenizer(source, return_tensors="pt", truncation=True)

        # Tokenize target as labels
        target_encoding = self._tokenizer(text_target=translation, return_tensors="pt", truncation=True)
        labels = target_encoding["input_ids"]

        # If the model forces a BOS token (e.g., a language code for NLLB/M2M100),
        # prepend it to the labels so the decoder has the correct context for scoring.
        forced_bos_token_id: Optional[int] = None
        if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
            forced_bos_token_id = self._model.generation_config.forced_bos_token_id
        if forced_bos_token_id is None and hasattr(self._model.config, "forced_bos_token_id"):
            forced_bos_token_id = self._model.config.forced_bos_token_id

        if forced_bos_token_id is not None and labels[0, 0].item() != forced_bos_token_id:
            forced_bos = torch.tensor([[forced_bos_token_id]], dtype=labels.dtype)
            labels = torch.cat([forced_bos, labels], dim=1)

        # Move tensors to the model's device
        device = next(self._model.parameters()).device
        input_ids = source_encoding["input_ids"].to(device)
        attention_mask = source_encoding["attention_mask"].to(device)
        labels_on_device = labels.to(device)

        # Run forward pass with teacher forcing to get logits at each position
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_on_device,
            )

        # Compute log probabilities from logits (use float32 for numerical precision)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]
        log_probs = F.log_softmax(logits.float(), dim=-1)

        label_ids = labels[0].tolist()
        tokens = [self._tokenizer.convert_ids_to_tokens(id) for id in label_ids]
        token_log_probs = [log_probs[0, i, id].item() for i, id in enumerate(label_ids)]

        # Get top-k alternative token IDs at each position for suggestions
        k = min(self._top_k_suggestions + 1, log_probs.shape[-1])
        top_k_ids = torch.topk(log_probs[0], k=k, dim=-1).indices.tolist()

        word_scores = self._group_tokens_into_words(tokens, token_log_probs, top_k_ids, label_ids)

        return ScoredTranslation(
            source=source,
            translation=translation,
            word_scores=word_scores,
        )

    def _is_word_initial(self, token: str) -> bool:
        """Return True if this subword token begins a new word."""
        return token.startswith("▁") or token.startswith("Ġ")

    def _decode_suggestion(self, token_id: int) -> Optional[str]:
        """Decode a token ID to a display string, returning None for special tokens."""
        if token_id in self._special_token_ids:
            return None
        token = self._tokenizer.convert_ids_to_tokens(token_id)
        if not token:
            return None
        # Strip SentencePiece (▁) or BPE (Ġ) word-initial markers
        if token.startswith("▁"):
            stripped = token.removeprefix("▁")
            return stripped if stripped else None
        if token.startswith("Ġ"):
            stripped = token.removeprefix("Ġ")
            return stripped if stripped else None
        # Strip BERT-style continuation marker
        if token.startswith("##"):
            return None
        return token

    def _group_tokens_into_words(
        self,
        tokens: List[str],
        token_log_probs: List[float],
        top_k_ids: List[List[int]],
        label_ids: List[int],
    ) -> List[WordScore]:
        """Group subword tokens into words and compute per-word scores."""
        word_scores: List[WordScore] = []
        current_token_scores: List[TokenScore] = []
        current_word_chars = ""
        current_first_top_k: List[int] = []

        for token, log_prob, position_top_k, label_id in zip(tokens, token_log_probs, top_k_ids, label_ids):
            # Skip special tokens (language codes, EOS, BOS, pad)
            if label_id in self._special_token_ids:
                if current_token_scores:
                    word_scores.append(
                        self._create_word_score(current_word_chars, current_token_scores, current_first_top_k)
                    )
                    current_token_scores = []
                    current_word_chars = ""
                    current_first_top_k = []
                continue

            # A word-initial token (starts with ▁ or Ġ) begins a new word
            starts_new_word = self._is_word_initial(token)
            if starts_new_word and current_token_scores:
                word_scores.append(
                    self._create_word_score(current_word_chars, current_token_scores, current_first_top_k)
                )
                current_token_scores = []
                current_word_chars = ""
                current_first_top_k = []

            # Record top-k alternatives only for the first subword of each word
            if not current_token_scores:
                current_first_top_k = position_top_k

            current_token_scores.append(TokenScore(token=token, log_prob=log_prob))

            # Append the token's characters to the current word (strip the word-initial marker)
            if token.startswith("▁"):
                current_word_chars += token.removeprefix("▁")
            elif token.startswith("Ġ"):
                current_word_chars += token.removeprefix("Ġ")
            elif token.startswith("##"):
                current_word_chars += token.removeprefix("##")
            else:
                current_word_chars += token

        # Finalize the last word
        if current_token_scores:
            word_scores.append(
                self._create_word_score(current_word_chars, current_token_scores, current_first_top_k)
            )

        return word_scores

    def _create_word_score(
        self,
        word: str,
        token_scores: List[TokenScore],
        first_token_top_k_ids: List[int],
    ) -> WordScore:
        """Build a WordScore, adding suggestions when the word is low-probability."""
        word_log_prob = sum(t.log_prob for t in token_scores)
        suggestions: List[str] = []

        if word_log_prob < self._low_prob_threshold:
            for top_id in first_token_top_k_ids:
                suggestion = self._decode_suggestion(top_id)
                if suggestion and suggestion.lower() != word.lower():
                    suggestions.append(suggestion)
                    if len(suggestions) >= self._top_k_suggestions:
                        break

        return WordScore(
            word=word,
            tokens=token_scores,
            suggestions=suggestions,
            low_prob_threshold=self._low_prob_threshold,
        )
