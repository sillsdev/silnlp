from pathlib import Path
from typing import List

import pytest

from silnlp.nmt.preprocessing import Preprocessing


class RecordingConfig:
    """Stands in for an experiment's config, recording the order it is asked to do things in."""

    def __init__(self, tokenize: bool = True, missing_files: List[Path] = []) -> None:
        self.data = {"tokenize": tokenize}
        self.inventory = RecordingInventory(missing_files)
        self.calls: List[str] = []
        self.vocabulary = RecordingVocabularyBuilder(self.calls)
        self.corpus_writer = RecordingCorpusWriter(self.calls)
        self.tokenizer = object()

    def create_vocabulary_builder(self) -> "RecordingVocabularyBuilder":
        return self.vocabulary

    def create_tokenizer(self):
        self.calls.append("create tokenizer")
        return self.tokenizer

    def create_corpus_writer(self, tokenizer, force_align: bool) -> "RecordingCorpusWriter":
        self.corpus_writer.tokenizer = tokenizer
        self.corpus_writer.force_align = force_align
        return self.corpus_writer


class RecordingInventory:
    def __init__(self, missing_files: List[Path]) -> None:
        self._missing_files = missing_files

    def missing_input_files(self) -> List[Path]:
        return self._missing_files


class RecordingVocabularyBuilder:
    def __init__(self, calls: List[str]) -> None:
        self._calls = calls
        self.stats = None

    def build(self, stats: bool = False) -> None:
        self._calls.append("build vocabulary")
        self.stats = stats


class RecordingCorpusWriter:
    def __init__(self, calls: List[str]) -> None:
        self._calls = calls
        self.tokenizer = None
        self.force_align = None
        self.stats = None

    def write(self, stats: bool) -> int:
        self._calls.append("write corpora")
        self.stats = stats
        return 0


def test_a_missing_corpus_file_stops_the_run_before_anything_is_written():
    config = RecordingConfig(missing_files=[Path("/data/en-BSB.txt")])

    with pytest.raises(RuntimeError, match="en-BSB.txt"):
        Preprocessing(config).run()

    assert config.calls == []


def test_every_missing_corpus_file_is_named(tmp_path):
    config = RecordingConfig(missing_files=[tmp_path / "one.txt", tmp_path / "two.txt"])

    with pytest.raises(RuntimeError, match="one.txt, .*two.txt"):
        Preprocessing(config).run()


def test_the_vocabulary_is_built_when_the_corpora_are_tokenized():
    config = RecordingConfig(tokenize=True)
    Preprocessing(config).run()

    assert "build vocabulary" in config.calls


def test_no_vocabulary_is_built_when_the_corpora_are_not_tokenized():
    config = RecordingConfig(tokenize=False)
    Preprocessing(config).run()

    assert "build vocabulary" not in config.calls
    assert "write corpora" in config.calls


def test_the_vocabulary_is_built_before_the_tokenizer_is_made():
    # Building the vocabulary changes the tokenizer, so the tokenizer has to be taken afterwards.
    config = RecordingConfig(tokenize=True)
    Preprocessing(config).run()

    assert config.calls == ["build vocabulary", "create tokenizer", "write corpora"]


def test_the_corpora_are_written_with_the_tokenizer_the_config_made():
    config = RecordingConfig()
    Preprocessing(config).run()

    assert config.corpus_writer.tokenizer is config.tokenizer


def test_the_request_for_statistics_reaches_both_the_vocabulary_and_the_corpora():
    config = RecordingConfig()
    Preprocessing(config).run(stats=True)

    assert config.vocabulary.stats is True
    assert config.corpus_writer.stats is True


def test_forcing_realignment_reaches_the_corpus_writer():
    config = RecordingConfig()
    Preprocessing(config).run(force_align=True)

    assert config.corpus_writer.force_align is True


def test_nothing_is_forced_or_measured_by_default():
    config = RecordingConfig()
    Preprocessing(config).run()

    assert config.corpus_writer.force_align is False
    assert config.corpus_writer.stats is False
