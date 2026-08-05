from typing import List, Sequence, Tuple

from silnlp.common.sentence_context import (
    CONTEXT_END_TOKEN,
    CONTEXT_START_TOKEN,
    CentralSegmentExtractor,
    build_context_windows,
    extract_central_segment,
    find_central_segment,
    find_central_token_span,
    iterate_context_windows,
)

SENTENCES = ["one", "two", "three", "four", "five", "six", "seven"]


def marked(sentence: str) -> str:
    return f"{CONTEXT_START_TOKEN} {sentence} {CONTEXT_END_TOKEN}"


def test_no_context_returns_sentences_unchanged():
    assert build_context_windows(SENTENCES, 0) == SENTENCES


def test_window_in_the_middle_has_context_on_both_sides():
    windows = build_context_windows(SENTENCES, 2)
    assert windows[3] == f"two three {marked('four')} five six"


def test_window_is_truncated_at_the_start_and_end_of_the_corpus():
    windows = build_context_windows(SENTENCES, 2)
    assert windows[0] == f"{marked('one')} two three"
    assert windows[1] == f"one {marked('two')} three four"
    assert windows[-1] == f"five six {marked('seven')}"
    assert windows[-2] == f"four five {marked('six')} seven"


def test_there_is_one_window_per_sentence():
    for context_size in range(0, 5):
        assert len(build_context_windows(SENTENCES, context_size)) == len(SENTENCES)


def test_context_size_of_one_takes_a_single_neighbor():
    assert build_context_windows(SENTENCES, 1)[3] == f"three {marked('four')} five"


def test_window_larger_than_the_corpus_takes_everything_available():
    assert build_context_windows(["a", "b"], 5) == [f"{marked('a')} b", f"a {marked('b')}"]


def test_windows_do_not_cross_group_boundaries():
    group_ids = [1, 1, 1, 2, 2, 2, 2]
    windows = build_context_windows(SENTENCES, 2, group_ids)
    assert windows[2] == f"one two {marked('three')}"
    assert windows[3] == f"{marked('four')} five six"


def test_a_repeated_group_id_starts_a_new_group():
    windows = build_context_windows(["a", "b", "c"], 1, [1, 2, 1])
    assert windows[0] == f"{marked('a')}"
    assert windows[2] == f"{marked('c')}"


def test_mismatched_group_ids_are_rejected():
    try:
        build_context_windows(SENTENCES, 1, [1, 2])
    except ValueError:
        return
    raise AssertionError("Expected a ValueError for group ids that do not correspond to the sentences.")


def test_empty_sentences_do_not_introduce_extra_whitespace():
    assert build_context_windows(["a", "", "c"], 1)[2] == f"{marked('c')}"
    assert build_context_windows(["a", "b", "c"], 1)[1] == f"a {marked('b')} c"


def test_whitespace_inside_sentences_is_collapsed():
    assert build_context_windows(["a  b", "c"], 1)[1] == f"a b {marked('c')}"


def rows(sentences: Sequence[str]) -> List[Tuple[str, str]]:
    return [(s, s.upper()) for s in sentences]


def test_streamed_windows_match_the_list_version():
    for context_size in range(0, 4):
        expected_src = build_context_windows(SENTENCES, context_size)
        expected_trg = build_context_windows([s.upper() for s in SENTENCES], context_size)
        streamed = list(iterate_context_windows(rows(SENTENCES), context_size))
        assert [row[0] for row, _ in streamed] == SENTENCES
        assert [windows[0] for _, windows in streamed] == expected_src
        assert [windows[1] for _, windows in streamed] == expected_trg


def test_streamed_windows_handle_a_corpus_shorter_than_the_window():
    streamed = list(iterate_context_windows(rows(["a", "b"]), 3))
    assert [windows[0] for _, windows in streamed] == [f"{marked('a')} b", f"a {marked('b')}"]


def test_streamed_windows_handle_an_empty_corpus():
    assert list(iterate_context_windows([], 2)) == []


def test_central_segment_is_recovered():
    assert find_central_segment(f"two three {marked('four')} five six") == "four"


def test_central_segment_is_none_when_the_markers_are_missing():
    assert find_central_segment("no markers here") is None
    assert find_central_segment(f"{CONTEXT_START_TOKEN} unterminated") is None
    assert find_central_segment(f"unopened {CONTEXT_END_TOKEN}") is None


def test_central_segment_falls_back_to_the_whole_text():
    assert extract_central_segment("no  markers here") == "no markers here"


def test_central_segment_of_an_empty_sentence_is_empty():
    assert find_central_segment(f"one {marked('')} three") == ""


def test_central_token_span_covers_the_tokens_between_the_markers():
    tokens = ["eng_Latn", "a", CONTEXT_START_TOKEN, "b", "c", CONTEXT_END_TOKEN, "d"]
    assert find_central_token_span(tokens) == (3, 5)
    assert tokens[3:5] == ["b", "c"]


def test_central_token_span_is_none_when_the_markers_are_missing():
    assert find_central_token_span(["a", "b"]) is None
    assert find_central_token_span([CONTEXT_START_TOKEN, "a"]) is None
    assert find_central_token_span([CONTEXT_END_TOKEN, CONTEXT_START_TOKEN]) is None


def test_extractor_counts_outputs_with_no_span():
    extractor = CentralSegmentExtractor()
    assert extractor.extract(f"a {marked('b')} c") == "b"
    assert extractor.extract("unmarked output") == "unmarked output"
    assert extractor.num_missing == 1


def test_extractor_ignores_blank_lines():
    extractor = CentralSegmentExtractor()
    assert extractor.extract("") == ""
    assert extractor.extract("   ") == "   "
    assert extractor.num_missing == 0


def test_disabled_extractor_passes_text_through():
    extractor = CentralSegmentExtractor(enabled=False)
    windowed = f"a {marked('b')} c"
    assert extractor.extract(windowed) == windowed
    assert extractor.num_missing == 0
