from types import SimpleNamespace
from typing import List, Union
from unittest.mock import patch

import torch

from silnlp.nmt.translation_scorer import (
    AbsoluteThresholdAnomalyDetector,
    ReplacementSuggestion,
    SpanScorer,
    SuggestionScorer,
    TargetSequenceProbabilities,
    TranslationScorer,
    WordTokenSpan,
    ZScoreAnomalyDetector,
)


class FakeTokenizer:
    all_special_ids = [0, 1, 2]
    eos_token_id = 1
    bos_token_id = 2


class FakeModel:
    def __init__(self):
        self.generation_config = SimpleNamespace(forced_bos_token_id=2)
        self.config = SimpleNamespace(forced_bos_token_id=None, decoder_start_token_id=2)

    def parameters(self):
        return iter([torch.zeros(1)])

    def eval(self):
        return None


def build_sequence(text: str, word_log_probs: List[Union[float, List[float]]]) -> TargetSequenceProbabilities:
    """Build a TargetSequenceProbabilities. A word's entry may be a single log-prob
    (one subword token) or a list of log-probs (a multi-token word)."""
    words = text.split() if text else []
    tokens: List[str] = []
    token_log_probs: List[float] = []
    spans: List[WordTokenSpan] = []
    index = 0
    for word, entry in zip(words, word_log_probs):
        sub_log_probs = entry if isinstance(entry, list) else [entry]
        start = index
        for sub in sub_log_probs:
            tokens.append(f"▁{word}")
            token_log_probs.append(sub)
            index += 1
        spans.append(WordTokenSpan(word, start, index))
    return TargetSequenceProbabilities(text=text, tokens=tokens, token_log_probs=token_log_probs, words=spans)


def make_scorer(**kwargs) -> TranslationScorer:
    return TranslationScorer(FakeModel(), FakeTokenizer(), **kwargs)


# --- TargetSequenceProbabilities -------------------------------------------


def test_sequence_normalizes_by_token_count_not_word_count():
    # "rare" is one word but three subword tokens; "the" is one word, one token.
    sequence = build_sequence("the rare", [-0.5, [-2.0, -2.0, -2.0]])
    assert sequence.log_prob(0, 2) == -6.5
    assert sequence.token_count(0, 2) == 4
    assert sequence.mean_token_log_prob == -6.5 / 4


# --- SpanScorer ------------------------------------------------------------


def test_span_score_uses_mean_per_token_and_windowed_right_context():
    base = build_sequence("bleu maison vite", [-1.0, -5.0, -0.5])
    scorer = SpanScorer(right_context_window=5)

    maison = scorer.score(base, 1, 2)
    assert maison.forward_log_prob == -5.0
    assert maison.token_count == 1
    assert maison.mean_token_log_prob == -5.0
    # Right context is the per-token mean of the following words (just "vite"), not a ratio.
    assert maison.right_context_log_prob == -0.5

    last = scorer.score(base, 2, 3)
    assert last.right_context_log_prob == 0.0  # no following words


# --- Anomaly detectors -----------------------------------------------------


def test_absolute_threshold_is_constant():
    detector = AbsoluteThresholdAnomalyDetector(-3.0)
    assert detector.threshold([-1.0, -2.0, -10.0]) == -3.0


def test_zscore_threshold_is_relative_to_sentence_spread():
    detector = ZScoreAnomalyDetector(z=1.0)
    values = [-1.0, -1.0, -1.0, -4.0]
    # mean = -1.75, pstdev = 1.299..., threshold = mean - 1*std
    threshold = detector.threshold(values)
    assert -3.1 < threshold < -3.0
    assert values[-1] < threshold  # the outlier is flagged
    assert values[0] > threshold  # the typical token is not


def test_zscore_flags_nothing_when_sentence_is_uniform_or_too_short():
    detector = ZScoreAnomalyDetector(z=1.5)
    assert detector.threshold([-1.0, -1.0, -1.0]) == float("-inf")
    assert detector.threshold([-1.0]) == float("-inf")


# --- SuggestionScorer ------------------------------------------------------


def test_suggestions_ranked_by_per_token_improvement_and_filtered():
    base = build_sequence("bleu maison vite", [-1.0, -5.0, -0.5])  # mean = -6.5/3
    span = SpanScorer().score(base, 1, 2)  # "maison"
    variants = {
        "bleu chien vite": build_sequence("bleu chien vite", [-1.0, -0.2, -0.5]),  # much better
        "bleu chat vite": build_sequence("bleu chat vite", [-1.0, -0.8, -0.5]),  # better
        "bleu maison vite": base,  # unchanged -> zero improvement, filtered
    }
    decoder = SimpleNamespace(score_target=lambda text: variants[text])
    scorer = SuggestionScorer(decoder, min_improvement=0.05, top_k=5)

    # "chien" and "chat" improve; the duplicate and the original-equal candidate are dropped.
    suggestions = scorer.score(base, span, ["chien", "chat", "chien", "maison"])

    assert [suggestion.phrase for suggestion in suggestions] == ["chien", "chat"]
    assert suggestions[0].improvement > suggestions[1].improvement
    assert all(isinstance(suggestion, ReplacementSuggestion) for suggestion in suggestions)


def test_suggestions_below_min_improvement_are_dropped():
    base = build_sequence("bleu maison vite", [-1.0, -5.0, -0.5])
    span = SpanScorer().score(base, 1, 2)
    variants = {"bleu chat vite": build_sequence("bleu chat vite", [-1.0, -4.9, -0.5])}
    decoder = SimpleNamespace(score_target=lambda text: variants[text])
    scorer = SuggestionScorer(decoder, min_improvement=0.5, top_k=5)

    assert scorer.score(base, span, ["chat"]) == []


# --- TranslationScorer end-to-end ------------------------------------------


def test_score_flags_low_probability_word_and_attaches_suggestions():
    scorer = make_scorer(anomaly_detector=AbsoluteThresholdAnomalyDetector(-3.0), max_phrase_words=2)
    sequences = {
        "bleu maison vite": build_sequence("bleu maison vite", [-1.0, -5.0, -0.5]),
        "bleu chien vite": build_sequence("bleu chien vite", [-1.0, -0.2, -0.5]),
    }

    with (
        patch.object(scorer._decoder, "set_source", return_value=None),
        patch.object(scorer._decoder, "score_target", side_effect=lambda text: sequences[text]),
        # score_targets_batch is used for variant scoring; score_target for base scoring.
        patch.object(scorer._decoder, "score_targets_batch", side_effect=lambda texts: [sequences[t] for t in texts]),
        # generate_batch replaces per-span generate calls.
        patch.object(scorer._candidate_generator, "generate_batch", side_effect=lambda prefixes: [["chien"]] * len(prefixes)),
    ):
        result = scorer.score("blue house quickly", "bleu maison vite")

    flagged = result.flagged_words
    assert [span.text for span in flagged] == ["maison"]
    assert [suggestion.phrase for suggestion in flagged[0].suggestions] == ["chien"]
    assert flagged[0].suggestions[0].improvement > 0
    # The whole-sentence aggregate is a plain forward log probability (<= 0).
    assert result.total_log_prob == -6.5
    assert result.mean_token_log_prob == -6.5 / 3


def test_score_does_not_flag_when_all_words_are_plausible():
    scorer = make_scorer(anomaly_detector=AbsoluteThresholdAnomalyDetector(-3.0), max_phrase_words=2)
    sequences = {"bleu maison": build_sequence("bleu maison", [-1.0, -2.0])}

    with (
        patch.object(scorer._decoder, "set_source", return_value=None),
        patch.object(scorer._decoder, "score_target", side_effect=lambda text: sequences[text]),
    ):
        result = scorer.score("blue house", "bleu maison")

    # Nothing flagged → generate_batch and score_targets_batch are never called.
    assert result.flagged_words == []
    assert result.flagged_phrases == []
