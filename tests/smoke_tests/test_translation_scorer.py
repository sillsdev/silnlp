"""Tests for the TranslationScorer class and related data structures."""
from unittest.mock import MagicMock, patch

import torch

from silnlp.nmt.translation_scorer import (
    DEFAULT_LOW_PROB_THRESHOLD,
    DEFAULT_TOP_K_SUGGESTIONS,
    ScoredTranslation,
    TokenScore,
    TranslationScorer,
    WordScore,
)

_TINY_MODEL_NAME = "hf-internal-testing/tiny-random-nllb"


def _make_mock_model_and_tokenizer():
    """Create a mock model and tokenizer for testing."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.modeling_outputs import Seq2SeqLMOutput

    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL_NAME)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "fra_Latn"

    # Create a minimal mock model that returns deterministic logits
    model = MagicMock()
    vocab_size = len(tokenizer)

    # Return logits shaped [1, seq_len, vocab_size] with deterministic values
    def mock_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
        seq_len = labels.shape[1] if labels is not None else 1
        logits = torch.zeros(1, seq_len, vocab_size)
        # Make the first token of the vocabulary highly probable except at position 0
        logits[:, :, 0] = -10.0
        logits[:, :, 1] = -1.0  # slightly low probability
        return Seq2SeqLMOutput(logits=logits)

    model.side_effect = None
    model.__call__ = MagicMock(side_effect=mock_forward)
    model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    model.eval = MagicMock()
    model.generation_config = MagicMock()
    model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("fra_Latn")
    model.config = MagicMock()
    model.config.forced_bos_token_id = None

    return model, tokenizer


class TestTokenScore:
    def test_prob_property(self):
        ts = TokenScore(token="▁hello", log_prob=-1.0)
        assert abs(ts.prob - 0.3679) < 1e-3

    def test_zero_log_prob(self):
        ts = TokenScore(token="▁hello", log_prob=0.0)
        assert ts.prob == 1.0


class TestWordScore:
    def test_log_prob_is_sum_of_token_log_probs(self):
        ws = WordScore(
            word="hello",
            tokens=[
                TokenScore(token="▁hel", log_prob=-0.5),
                TokenScore(token="lo", log_prob=-0.3),
            ],
        )
        assert abs(ws.log_prob - (-0.8)) < 1e-6

    def test_is_low_probability_default_threshold(self):
        low = WordScore(word="low", tokens=[TokenScore(token="▁low", log_prob=-5.0)])
        high = WordScore(word="high", tokens=[TokenScore(token="▁high", log_prob=-1.0)])
        assert low.is_low_probability
        assert not high.is_low_probability

    def test_is_low_probability_custom_threshold(self):
        ws = WordScore(
            word="test",
            tokens=[TokenScore(token="▁test", log_prob=-2.0)],
            low_prob_threshold=-1.0,
        )
        assert ws.is_low_probability


class TestScoredTranslation:
    def test_sequence_log_prob(self):
        st = ScoredTranslation(
            source="hello",
            translation="bonjour",
            word_scores=[
                WordScore(word="bonjour", tokens=[TokenScore(token="▁bonjour", log_prob=-0.5)]),
            ],
        )
        assert abs(st.sequence_log_prob - (-0.5)) < 1e-6

    def test_low_probability_words(self):
        st = ScoredTranslation(
            source="hello world",
            translation="bonjour monde",
            word_scores=[
                WordScore(word="bonjour", tokens=[TokenScore(token="▁bonjour", log_prob=-1.0)]),
                WordScore(word="monde", tokens=[TokenScore(token="▁monde", log_prob=-5.0)]),
            ],
        )
        low = st.low_probability_words
        assert len(low) == 1
        assert low[0].word == "monde"


class TestTranslationScorer:
    def test_score_returns_scored_translation(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = TranslationScorer(model, tokenizer)
        result = scorer.score("hello world", "bonjour monde")
        assert isinstance(result, ScoredTranslation)
        assert result.source == "hello world"
        assert result.translation == "bonjour monde"
        assert isinstance(result.word_scores, list)
        assert len(result.word_scores) > 0

    def test_each_word_score_has_tokens(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = TranslationScorer(model, tokenizer)
        result = scorer.score("hello world", "bonjour monde")
        for ws in result.word_scores:
            assert isinstance(ws, WordScore)
            assert len(ws.tokens) > 0
            for ts in ws.tokens:
                assert isinstance(ts, TokenScore)

    def test_low_prob_words_get_suggestions(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        # Use a threshold of 0.0 so all words are flagged as low-probability
        scorer = TranslationScorer(model, tokenizer, low_prob_threshold=0.0)
        result = scorer.score("hello world", "bonjour monde")
        for ws in result.word_scores:
            assert ws.is_low_probability
            # Every flagged word should have suggestions (up to top_k)
            assert len(ws.suggestions) <= DEFAULT_TOP_K_SUGGESTIONS

    def test_custom_top_k(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = TranslationScorer(model, tokenizer, low_prob_threshold=0.0, top_k_suggestions=2)
        result = scorer.score("hello", "bonjour")
        for ws in result.word_scores:
            assert len(ws.suggestions) <= 2

    def test_forced_bos_prepended_to_labels(self):
        """Verify that forced_bos_token_id is prepended to labels when not already present."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = TranslationScorer(model, tokenizer)

        # Capture the labels passed to the model
        captured_labels = []
        original_call = model.__call__.side_effect

        def capturing_call(**kwargs):
            captured_labels.append(kwargs.get("labels"))
            return original_call(**kwargs)

        model.__call__.side_effect = capturing_call
        scorer.score("hello", "bonjour")

        assert len(captured_labels) == 1
        labels = captured_labels[0]
        # The first token should be the forced BOS (language code)
        forced_bos = tokenizer.convert_tokens_to_ids("fra_Latn")
        assert labels[0, 0].item() == forced_bos


class TestWordGrouping:
    def test_sentencepiece_tokens_grouped_correctly(self):
        """Test that SentencePiece tokens (▁ prefix) are grouped into words."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = TranslationScorer(model, tokenizer)

        # Mock _group_tokens_into_words directly to test grouping logic
        tokens = ["▁hello", "▁world", "!"]
        token_log_probs = [-1.0, -2.0, -0.5]
        label_ids = [101, 102, 103]  # fake IDs, not in special_token_ids
        top_k_ids = [[1, 2, 3, 4, 5, 6]] * 3

        # Temporarily clear special token IDs to prevent any skipping
        scorer._special_token_ids = set()
        words = scorer._group_tokens_into_words(tokens, token_log_probs, top_k_ids, label_ids)

        # "hello" is one word, "world" + "!" are two words
        # "!" does not start with ▁, so it continues "world" → "world!"
        assert len(words) == 2
        assert words[0].word == "hello"
        assert words[1].word == "world!"

    def test_continuation_tokens_merged(self):
        """Test that tokens without ▁ are merged with the previous word."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = TranslationScorer(model, tokenizer)
        scorer._special_token_ids = set()

        tokens = ["▁walk", "ing"]
        token_log_probs = [-0.5, -0.3]
        label_ids = [101, 102]
        top_k_ids = [[1, 2, 3, 4, 5, 6]] * 2

        words = scorer._group_tokens_into_words(tokens, token_log_probs, top_k_ids, label_ids)
        assert len(words) == 1
        assert words[0].word == "walking"
        assert len(words[0].tokens) == 2

    def test_special_tokens_skipped(self):
        """Special tokens (EOS, BOS, lang codes) should not appear as words."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = TranslationScorer(model, tokenizer)

        # Get some real special token IDs
        eos_id = tokenizer.eos_token_id
        special_ids = {eos_id}
        scorer._special_token_ids = special_ids

        tokens = ["▁hello", tokenizer.eos_token]
        token_log_probs = [-1.0, 0.0]
        label_ids = [101, eos_id]
        top_k_ids = [[1, 2, 3, 4, 5, 6]] * 2

        words = scorer._group_tokens_into_words(tokens, token_log_probs, top_k_ids, label_ids)
        # EOS token should be skipped; only "hello" remains
        assert len(words) == 1
        assert words[0].word == "hello"
