import json
from collections import Counter
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from silnlp.nmt.vocab_extension import (
    build_extended_bpe_tokenizer,
    find_missing_characters,
    learn_bpe_merges,
    normalize_merges,
    split_added_token_counts,
    tokenize_words,
)

NLLB_TOKENIZER_PATH = Path(__file__).parents[2] / "silnlp" / "assets" / "tokenizers" / "facebook" / "nllb-200"


def base_tokenizer_json(vocab, merges):
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
        "post_processor": None,
        "decoder": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "vocab": vocab,
            "merges": merges,
        },
    }


# --- learn_bpe_merges ---------------------------------------------------------------------------


def test_learn_bpe_merges_picks_the_most_frequent_pair():
    counts = Counter({("a", "b"): 3, ("c", "d"): 9})
    merges, tokens, _ = learn_bpe_merges(counts, 1, vocab=set())
    assert merges == [("c", "d")]
    assert tokens == ["cd"]


def test_learn_bpe_merges_returns_exactly_the_requested_number_of_merges():
    counts = Counter({("a", "b", "c", "d", "e"): 10})
    merges, tokens, _ = learn_bpe_merges(counts, 3, vocab=set())
    assert len(merges) == 3
    assert len(tokens) == 3


def test_learn_bpe_merges_returns_one_new_token_per_merge():
    counts = Counter({("a", "b", "c"): 5, ("d", "e"): 4})
    merges, tokens, counts_out = learn_bpe_merges(counts, 3, vocab=set())
    assert len(tokens) == len(merges) == len(counts_out)
    assert len(set(tokens)) == len(tokens)
    for (left, right), token in zip(merges, tokens):
        assert left + right == token


def test_learn_bpe_merges_stops_early_when_the_corpus_is_exhausted():
    counts = Counter({("a", "b"): 5})
    merges, tokens, _ = learn_bpe_merges(counts, 8, vocab=set())
    assert merges == [("a", "b")]
    assert tokens == ["ab"]


def test_learn_bpe_merges_respects_min_frequency():
    counts = Counter({("a", "b"): 1})
    merges, tokens, _ = learn_bpe_merges(counts, 4, vocab=set(), min_frequency=2)
    assert merges == []
    assert tokens == []


def test_learn_bpe_merges_breaks_ties_deterministically():
    forwards = Counter({("a", "b"): 5, ("c", "d"): 5})
    backwards = Counter({("c", "d"): 5, ("a", "b"): 5})
    assert learn_bpe_merges(forwards, 1, vocab=set())[0] == [("a", "b")]
    assert learn_bpe_merges(backwards, 1, vocab=set())[0] == [("a", "b")]


def test_learn_bpe_merges_skips_pairs_already_in_the_vocabulary():
    counts = Counter({("a", "b"): 9, ("c", "d"): 4})
    merges, tokens, _ = learn_bpe_merges(counts, 1, vocab={"ab"})
    assert merges == [("c", "d")]
    assert tokens == ["cd"]


def test_learn_bpe_merges_never_emits_a_duplicate_token():
    # Both "ab" + "c" and "a" + "bc" concatenate to "abc"; only the first may be added.
    counts = Counter({("a", "b", "c"): 10, ("a", "bc"): 9})
    merges, tokens, _ = learn_bpe_merges(counts, 3, vocab=set())
    assert tokens == ["ab", "abc"]
    assert ("a", "bc") not in merges


def test_learn_bpe_merges_handles_overlapping_pairs():
    counts = Counter({("a", "a", "a"): 5})
    merges, tokens, _ = learn_bpe_merges(counts, 2, vocab=set())
    assert merges == [("a", "a"), ("aa", "a")]
    assert tokens == ["aa", "aaa"]


def test_learn_bpe_merges_does_not_merge_across_words():
    counts = Counter({("a",): 100, ("b",): 100})
    merges, tokens, _ = learn_bpe_merges(counts, 4, vocab=set())
    assert merges == []
    assert tokens == []


def test_learn_bpe_merges_honours_forbidden_pairs():
    counts = Counter({("a", "b"): 9, ("c", "d"): 4})
    merges, _, _ = learn_bpe_merges(counts, 1, vocab=set(), forbidden_pairs=frozenset({("a", "b")}))
    assert merges == [("c", "d")]


def test_learn_bpe_merges_returns_firing_counts_matching_corpus_frequency():
    counts = Counter({("a", "b"): 7, ("a", "b", "z"): 2})
    _, _, firing_counts = learn_bpe_merges(counts, 1, vocab=set())
    assert firing_counts == [9]


# --- split_added_token_counts -------------------------------------------------------------------


def test_split_added_token_counts_attributes_each_side_its_own_tokens():
    assert split_added_token_counts([], ["a", "b"], ["c"]) == (2, 1)


def test_split_added_token_counts_divides_shared_tokens_evenly():
    assert split_added_token_counts(["a", "b", "c", "d"], [], []) == (2, 2)


def test_split_added_token_counts_gives_the_odd_shared_token_to_the_target():
    assert split_added_token_counts(["a", "b", "c"], [], []) == (1, 2)


def test_split_added_token_counts_adds_shared_and_per_side_tokens():
    assert split_added_token_counts(["a", "b"], ["c"], ["d", "e"]) == (2, 3)


# --- normalize_merges ---------------------------------------------------------------------------


def test_normalize_merges_accepts_space_joined_strings():
    assert normalize_merges(["a n", "▁ m"]) == [["a", "n"], ["▁", "m"]]


def test_normalize_merges_accepts_pairs():
    assert normalize_merges([["a", "n"], ("▁", "m")]) == [["a", "n"], ["▁", "m"]]


def test_normalize_merges_rejects_malformed_entries():
    with pytest.raises(ValueError):
        normalize_merges(["a b c"])
    with pytest.raises(ValueError):
        normalize_merges([42])


# --- build_extended_bpe_tokenizer ------------------------------------------------------------------


def test_build_extended_bpe_tokenizer_appends_merges_at_the_end():
    vocab = {"<unk>": 0, "▁": 1, "a": 2, "b": 3, "c": 4, "ab": 5}
    data = base_tokenizer_json(vocab, [["a", "b"]])
    before = Tokenizer.from_str(json.dumps(data))
    assert [t.value for t in before.model.tokenize("abc")] == ["ab", "c"]

    after, _ = build_extended_bpe_tokenizer(data, ["abc"], [("ab", "c")])
    assert [t.value for t in after.model.tokenize("abc")] == ["abc"]
    # A word the base rules already resolved is untouched by the appended rule.
    assert [t.value for t in after.model.tokenize("ab")] == ["ab"]


def test_build_extended_bpe_tokenizer_assigns_sequential_ids():
    vocab = {"<unk>": 0, "▁": 1, "a": 2, "b": 3}
    _, data = build_extended_bpe_tokenizer(base_tokenizer_json(vocab, []), ["x", "y"], [])
    assert data["model"]["vocab"]["x"] == 4
    assert data["model"]["vocab"]["y"] == 5


def test_build_extended_bpe_tokenizer_is_loadable_when_base_merges_are_strings():
    # Mixing space-joined strings and pairs in one file fails to deserialize.
    vocab = {"<unk>": 0, "▁": 1, "a": 2, "b": 3, "c": 4, "ab": 5, "abc": 6}
    tokenizer, data = build_extended_bpe_tokenizer(base_tokenizer_json(vocab, ["a b"]), [], [("ab", "c")])
    assert all(isinstance(merge, list) for merge in data["model"]["merges"])
    assert [t.value for t in tokenizer.model.tokenize("abc")] == ["abc"]


def test_build_extended_bpe_tokenizer_requires_merge_results_in_vocab():
    vocab = {"<unk>": 0, "▁": 1, "a": 2, "b": 3}
    with pytest.raises(Exception):
        build_extended_bpe_tokenizer(base_tokenizer_json(vocab, []), [], [("a", "b")])


# --- tokenize_words ----------------------------------------------------------------------------


def test_tokenize_words_rejects_words_containing_unk():
    vocab = {"<unk>": 0, "▁": 1, "a": 2}
    tokenizer, _ = build_extended_bpe_tokenizer(base_tokenizer_json(vocab, []), [], [])
    tokenized, rejected = tokenize_words(tokenizer, Counter({"▁a": 3, "▁z": 4}))
    assert tokenized == Counter({("▁", "a"): 3})
    assert rejected == Counter({"▁z": 4})


# --- find_missing_characters --------------------------------------------------------------------------


def test_find_missing_characters_excludes_vocab_and_whitespace():
    counts = Counter({"▁abç": 1, "▁d": 1})
    assert find_missing_characters(counts, {"▁", "a", "b"}) == ["d", "ç"]


# --- integration against the real NLLB tokenizer ------------------------------------------------


@pytest.mark.skipif(not (NLLB_TOKENIZER_PATH / "tokenizer.json").is_file(), reason="NLLB tokenizer asset not present")
def test_nllb_first_merges_change_only_the_target_words():
    base_json = json.loads((NLLB_TOKENIZER_PATH / "tokenizer.json").read_text(encoding="utf-8"))
    base = Tokenizer.from_str(json.dumps(base_json))

    control = "the quick brown fox jumped"
    control_before = base.encode(control, add_special_tokens=False).tokens

    words = Counter({"▁wɛnɛ": 20, "▁taako": 15, "▁Yesu": 10})
    missing_chars = find_missing_characters(words, set(base_json["model"]["vocab"]))
    staged, staged_json = build_extended_bpe_tokenizer(base_json, missing_chars, [])
    tokenized, rejected = tokenize_words(staged, words)
    assert not rejected

    base_vocab_size = len(base_json["model"]["vocab"])
    merges, tokens, _ = learn_bpe_merges(tokenized, 3, set(staged.get_vocab()))
    base_pairs = {tuple(m) for m in normalize_merges(base_json["model"]["merges"])}
    assert not base_pairs.intersection(merges)

    extended, extended_json = build_extended_bpe_tokenizer(staged_json, tokens, merges)
    assert len(extended_json["model"]["vocab"]) == base_vocab_size + len(missing_chars) + len(merges)
    # Every learned token is now emitted as a single token.
    for token in tokens:
        assert [t.value for t in extended.model.tokenize(token)] == [token]
    # Text the base tokenizer already handled is segmented identically.
    assert extended.encode(control, add_special_tokens=False).tokens == control_before
