import json
from pathlib import Path

from silnlp.nmt.vocabulary import TokenizerVocabularyFile, VocabularyExtension


def extension(tokens=(), source=(), target=(), shared=False, trained=()) -> VocabularyExtension:
    return VocabularyExtension(list(tokens), list(source), list(target), list(trained), shared)


class RecordingVocabulary:
    """Stands in for the saved tokenizer, recording what it was asked to add."""

    def __init__(self) -> None:
        self.added = None

    def add(self, tokens, trained_tokenizers) -> None:
        self.added = (tokens, trained_tokenizers)


def test_nothing_is_added_when_no_tokens_are_missing():
    vocabulary = RecordingVocabulary()
    extension().add_to(vocabulary)

    assert vocabulary.added is None


def test_the_missing_tokens_are_added_with_whatever_trained_them():
    vocabulary = RecordingVocabulary()
    extension(tokens=["a", "b"], trained=["trained"]).add_to(vocabulary)

    assert vocabulary.added == (["a", "b"], ["trained"])


def test_each_side_is_counted_separately():
    assert extension(source=["a", "b"], target=["c"]).counts_by_side() == [["Source", 2], ["Target", 1]]


def test_a_shared_vocabulary_splits_its_tokens_evenly_between_the_sides():
    counts = extension(tokens=["a", "b", "c", "d"], shared=True).counts_by_side()

    assert counts == [["Source", 2], ["Target", 2]]


def test_a_shared_vocabulary_gives_the_odd_token_to_the_target():
    counts = extension(tokens=["a", "b", "c"], shared=True).counts_by_side()

    assert counts == [["Source", 1], ["Target", 2]]


def test_an_empty_extension_counts_nothing_on_either_side():
    assert extension().counts_by_side() == [["Source", 0], ["Target", 0]]


def byte_pair_file(tmp_path: Path, vocab: dict, merges: list) -> Path:
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab, "merges": merges}}), encoding="utf-8")
    return path


def unigram_file(tmp_path: Path, vocab: list) -> Path:
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"type": "Unigram", "vocab": vocab}}), encoding="utf-8")
    return path


def model_of(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["model"]


def test_new_byte_pair_tokens_are_numbered_from_the_end_of_the_vocabulary(tmp_path):
    path = byte_pair_file(tmp_path, {"a": 0, "b": 1}, [])
    TokenizerVocabularyFile(path).add(["c", "d"], [])

    assert model_of(path)["vocab"] == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_trained_byte_pair_merges_take_precedence_over_the_originals(tmp_path):
    # Merges are applied in order, so the trained ones have to come first to be reachable.
    path = byte_pair_file(tmp_path, {"a": 0}, ["base merge"])
    TokenizerVocabularyFile(path).add([], [{"model": {"merges": ["trained merge"]}}])

    assert model_of(path)["merges"] == ["trained merge", "base merge"]


def test_several_trained_tokenizers_stack_their_merges_in_reverse_order(tmp_path):
    path = byte_pair_file(tmp_path, {"a": 0}, ["base"])
    TokenizerVocabularyFile(path).add(
        [], [{"model": {"merges": ["first"]}}, {"model": {"merges": ["second"]}}]
    )

    assert model_of(path)["merges"] == ["second", "first", "base"]


def test_new_unigram_tokens_are_appended_with_a_fixed_score(tmp_path):
    path = unigram_file(tmp_path, [["a", -1.0]])
    TokenizerVocabularyFile(path).add(["b"], [])

    assert model_of(path)["vocab"] == [["a", -1.0], ["b", -18]]


def test_a_trained_unigram_vocabulary_is_appended_instead_of_the_bare_tokens(tmp_path):
    path = unigram_file(tmp_path, [["a", -1.0]])
    TokenizerVocabularyFile(path).add(["ignored"], [{"model": {"vocab": [["b", -2.0]]}}])

    assert model_of(path)["vocab"] == [["a", -1.0], ["b", -2.0]]


def test_a_trained_token_the_base_already_has_keeps_the_base_score(tmp_path):
    path = unigram_file(tmp_path, [["a", -1.0]])
    TokenizerVocabularyFile(path).add([], [{"model": {"vocab": [["a", -9.0], ["b", -2.0]]}}])

    assert model_of(path)["vocab"] == [["a", -1.0], ["b", -2.0]]


def test_a_later_trained_vocabulary_is_compared_against_what_the_earlier_one_added(tmp_path):
    path = unigram_file(tmp_path, [["a", -1.0]])
    TokenizerVocabularyFile(path).add(
        [], [{"model": {"vocab": [["b", -2.0]]}}, {"model": {"vocab": [["b", -9.0], ["c", -3.0]]}}]
    )

    assert model_of(path)["vocab"] == [["a", -1.0], ["b", -2.0], ["c", -3.0]]


def test_a_model_of_an_unknown_type_is_left_alone(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"type": "WordPiece", "vocab": {"a": 0}}}), encoding="utf-8")

    TokenizerVocabularyFile(path).add(["b"], [])

    assert model_of(path)["vocab"] == {"a": 0}
