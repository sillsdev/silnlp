from silnlp.nmt.translation_scorer_webapp import _FlagCandidate, _select_non_overlapping_flags, _word_char_spans


def test_word_char_spans_returns_non_whitespace_ranges():
    assert _word_char_spans("bleu   maison vite") == [(0, 4), (7, 13), (14, 18)]


def test_select_non_overlapping_flags_prefers_longer_phrases_at_same_start():
    candidates = [
        _FlagCandidate("bleu", 0, 1, -4.0, [], "word"),
        _FlagCandidate("bleu maison", 0, 2, -3.8, [], "phrase"),
        _FlagCandidate("maison", 1, 2, -3.6, [], "word"),
        _FlagCandidate("vite", 2, 3, -4.5, [], "word"),
    ]

    selected = _select_non_overlapping_flags(candidates)

    assert [(candidate.text, candidate.span_start, candidate.span_end) for candidate in selected] == [
        ("bleu maison", 0, 2),
        ("vite", 2, 3),
    ]
