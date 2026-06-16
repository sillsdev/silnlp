from types import SimpleNamespace
from unittest.mock import patch

from silnlp.nmt.translation_scorer import PhraseScore, SequenceScore, TokenScore, TranslationScorer, WordSpan


class FakeTokenizer:
    all_special_ids = [0, 1, 2]
    eos_token_id = 1
    bos_token_id = 2


class FakeModel:
    def __init__(self):
        self.generation_config = SimpleNamespace(forced_bos_token_id=2)
        self.config = SimpleNamespace(forced_bos_token_id=None, decoder_start_token_id=2)

    def eval(self):
        return None


def build_sequence(text: str, word_log_probs: list[float]) -> SequenceScore:
    words = text.split() if text else []
    return SequenceScore(
        text=text,
        label_ids=list(range(10, 10 + len(words))),
        tokens=[f"▁{word}" for word in words],
        token_log_probs=word_log_probs,
        words=[WordSpan(word, index, index + 1) for index, word in enumerate(words)],
    )


def make_scorer(**kwargs) -> TranslationScorer:
    return TranslationScorer(FakeModel(), FakeTokenizer(), min_suggestion_improvement=0.0, **kwargs)


def test_contextual_word_scoring_uses_right_context_delta():
    scorer = make_scorer(low_prob_threshold=-3.0, max_phrase_words=2)
    scores = {
        "bleu maison vite": build_sequence("bleu maison vite", [-1.0, -5.0, -0.5]),
        "maison vite": build_sequence("maison vite", [-0.5, -0.5]),
        "bleu vite": build_sequence("bleu vite", [-1.0, -0.2]),
        "vite": build_sequence("vite", [-0.2]),
    }

    with (
        patch.object(scorer, "_encode_source", return_value={}),
        patch.object(scorer, "_score_translation_text", side_effect=lambda _ctx, text, _cache: scores[text]),
        patch.object(scorer, "_generate_candidate_phrases", return_value=[]),
    ):
        result = scorer.score("blue house quickly", "bleu maison vite")

    bleu = next(word for word in result.word_scores if word.word == "bleu")
    maison = next(word for word in result.word_scores if word.word == "maison")

    assert bleu.forward_log_prob == -1.0
    assert bleu.right_context_log_prob == -4.5
    assert bleu.contextual_log_prob == -5.5
    assert bleu.is_low_probability
    assert maison.contextual_log_prob < maison.forward_log_prob


def test_phrase_scores_are_length_normalized_for_variable_length_phrases():
    scorer = make_scorer(low_prob_threshold=-3.0, max_phrase_words=3)
    scores = {
        "bleu maison vite": build_sequence("bleu maison vite", [-1.0, -5.0, -0.5]),
        "maison vite": build_sequence("maison vite", [-0.5, -0.5]),
        "bleu vite": build_sequence("bleu vite", [-1.0, -0.2]),
        "vite": build_sequence("vite", [-0.2]),
        "bleu maison": build_sequence("bleu maison", [-1.0, -0.3]),
    }

    with (
        patch.object(scorer, "_encode_source", return_value={}),
        patch.object(scorer, "_score_translation_text", side_effect=lambda _ctx, text, _cache: scores[text]),
        patch.object(scorer, "_generate_candidate_phrases", return_value=[]),
    ):
        result = scorer.score("blue house quickly", "bleu maison vite")

    phrase = next(score for score in result.phrase_scores if score.phrase == "bleu maison")
    assert isinstance(phrase, PhraseScore)
    assert phrase.forward_log_prob == -6.0
    assert phrase.right_context_log_prob == -0.3
    assert phrase.contextual_log_prob == -6.3
    assert phrase.normalized_log_prob == -3.15
    assert phrase.is_low_probability


def test_replacement_suggestions_are_rescored_with_right_context_and_can_change_length():
    scorer = make_scorer(low_prob_threshold=-3.0, top_k_suggestions=3, max_phrase_words=2)
    scores = {
        "bleu maison vite": build_sequence("bleu maison vite", [-1.0, -5.0, -0.5]),
        "maison vite": build_sequence("maison vite", [-0.5, -0.5]),
        "blue maison vite": build_sequence("blue maison vite", [-0.8, -0.2, -0.5]),
        "blue house maison vite": build_sequence("blue house maison vite", [-0.7, -0.3, -0.3, -0.4]),
        "azure maison vite": build_sequence("azure maison vite", [-1.5, -0.6, -0.8]),
        "bleu vite": build_sequence("bleu vite", [-1.0, -0.2]),
        "blue vite": build_sequence("blue vite", [-0.7, -0.2]),
        "blue house vite": build_sequence("blue house vite", [-0.7, -0.4, -0.1]),
        "azure vite": build_sequence("azure vite", [-1.3, -0.3]),
        "vite": build_sequence("vite", [-0.2]),
    }

    with (
        patch.object(scorer, "_encode_source", return_value={}),
        patch.object(scorer, "_score_translation_text", side_effect=lambda _ctx, text, _cache: scores[text]),
        patch.object(scorer, "_generate_candidate_phrases", return_value=["blue", "blue house", "azure"]),
    ):
        result = scorer.score("blue house quickly", "bleu maison vite")

    bleu = next(word for word in result.word_scores if word.word == "bleu")
    phrase = next(score for score in result.phrase_scores if score.phrase == "bleu maison")

    assert [suggestion.phrase for suggestion in bleu.suggestions] == ["blue house", "blue", "azure"]
    assert bleu.suggestions[0].normalized_log_prob > bleu.suggestions[1].normalized_log_prob
    assert bleu.suggestions[0].improvement > 0

    assert [suggestion.phrase for suggestion in phrase.suggestions] == ["blue house", "blue", "azure"]
    assert phrase.suggestions[0].phrase == "blue house"
    assert phrase.suggestions[0].normalized_log_prob > phrase.normalized_log_prob


def test_sequence_log_probability_still_tracks_forward_sentence_score():
    scorer = make_scorer(low_prob_threshold=-10.0, max_phrase_words=1)
    scores = {
        "bleu maison": build_sequence("bleu maison", [-1.0, -2.0]),
        "maison": build_sequence("maison", [-0.5]),
    }

    with (
        patch.object(scorer, "_encode_source", return_value={}),
        patch.object(scorer, "_score_translation_text", side_effect=lambda _ctx, text, _cache: scores[text]),
        patch.object(scorer, "_generate_candidate_phrases", return_value=[]),
    ):
        result = scorer.score("blue house", "bleu maison")

    assert result.sequence_log_prob == -3.0
    assert all(isinstance(token, TokenScore) for word in result.word_scores for token in word.tokens)
