from silnlp.nmt.translation_scorer_webapp import (
    _FlagCandidate,
    _select_max_improvement_flags,
    _word_char_spans,
)


def _with_improvement(text: str, start: int, end: int, log_prob: float, improvement: float, kind: str) -> _FlagCandidate:
    suggestions = [{"phrase": "x", "mean_token_log_prob": -1.0, "improvement": improvement}] if improvement > 0 else []
    return _FlagCandidate(text, start, end, log_prob, suggestions, kind)


def test_word_char_spans_returns_non_whitespace_ranges():
    assert _word_char_spans("bleu   maison vite") == [(0, 4), (7, 13), (14, 18)]


def test_selects_nothing_when_no_candidate_has_suggestions():
    candidates = [
        _FlagCandidate("bleu", 0, 1, -4.0, [], "word"),
        _FlagCandidate("maison", 1, 2, -3.6, [], "word"),
    ]
    assert _select_max_improvement_flags(candidates) == []


def test_selects_single_candidate_with_positive_improvement():
    candidates = [_with_improvement("bleu", 0, 1, -4.0, 0.5, "word")]
    selected = _select_max_improvement_flags(candidates)
    assert [c.text for c in selected] == ["bleu"]


def test_prefers_higher_weight_non_overlapping_subset():
    # "bleu maison" (words 0-2, improvement 3.0, weight=3.0*2=6.0) overlaps with
    # "bleu" (0-1, weight=2.0) and "maison" (1-2, weight=1.5).
    # Optimal: "bleu maison" (6.0) + "vite" (4.0) = 10.0
    # vs "bleu" (2.0) + "maison" (1.5) + "vite" (4.0) = 7.5
    candidates = [
        _with_improvement("bleu", 0, 1, -4.0, 2.0, "word"),
        _with_improvement("bleu maison", 0, 2, -3.8, 3.0, "phrase"),
        _with_improvement("maison", 1, 2, -3.6, 1.5, "word"),
        _with_improvement("vite", 2, 3, -4.5, 4.0, "word"),
    ]
    selected = _select_max_improvement_flags(candidates)
    assert [c.text for c in selected] == ["bleu maison", "vite"]


def test_chooses_two_words_over_one_phrase_when_words_have_higher_total():
    # "bleu maison" (0-2, weight=0.5*2=1.0) vs "bleu"(0-1, weight=1.5) + "maison"(1-2, weight=1.5) = 3.0
    candidates = [
        _with_improvement("bleu", 0, 1, -4.0, 1.5, "word"),
        _with_improvement("bleu maison", 0, 2, -3.8, 0.5, "phrase"),
        _with_improvement("maison", 1, 2, -3.6, 1.5, "word"),
    ]
    selected = _select_max_improvement_flags(candidates)
    assert {c.text for c in selected} == {"bleu", "maison"}


def test_skips_zero_improvement_candidate_that_would_block_better_one():
    # "bleu maison" overlaps with "maison" (improvement=2.0).
    # "bleu maison" has no suggestions → weight 0, so it is excluded.
    candidates = [
        _FlagCandidate("bleu maison", 0, 2, -3.8, [], "phrase"),  # no suggestions
        _with_improvement("maison", 1, 2, -3.6, 2.0, "word"),
    ]
    selected = _select_max_improvement_flags(candidates)
    assert [c.text for c in selected] == ["maison"]
