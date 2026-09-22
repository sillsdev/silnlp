from pathlib import Path
from typing import List

import pytest

from silnlp.nmt.preprocessor import Preprocessor


class RecordingInventory:
    def __init__(self, missing_files: List[Path] = []) -> None:
        self._missing_files = missing_files

    def missing_input_files(self) -> List[Path]:
        return self._missing_files


class RecordingVocabularyBuilder:
    def __init__(self, calls: List[str]) -> None:
        self._calls = calls
        self.stats = None

    def build(self, stats: bool = False):
        self._calls.append("build vocabulary")
        self.stats = stats
        return None


class RecordingDataSetWriter:
    def __init__(self, calls: List[str]) -> None:
        self._calls = calls
        self.stats = None

    def write(self, stats: bool) -> int:
        self._calls.append("write data sets")
        self.stats = stats
        return 0


def preprocessor_for(missing_files: List[Path] = []):
    calls: List[str] = []
    vocabulary = RecordingVocabularyBuilder(calls)
    data_sets = RecordingDataSetWriter(calls)
    return Preprocessor(RecordingInventory(missing_files), vocabulary, data_sets), calls, vocabulary, data_sets


def test_a_missing_corpus_file_stops_the_run_before_anything_is_written():
    preprocessor, calls, _, _ = preprocessor_for([Path("/data/en-BSB.txt")])

    with pytest.raises(RuntimeError, match="en-BSB.txt"):
        preprocessor.run()

    assert calls == []


def test_every_missing_corpus_file_is_named(tmp_path):
    preprocessor, _, _, _ = preprocessor_for([tmp_path / "one.txt", tmp_path / "two.txt"])

    with pytest.raises(RuntimeError, match="one.txt, .*two.txt"):
        preprocessor.run()


def test_the_vocabulary_is_settled_before_the_data_sets_are_written():
    # The data set writers tokenize with the vocabulary this step produces.
    preprocessor, calls, _, _ = preprocessor_for()
    preprocessor.run()

    assert calls == ["build vocabulary", "write data sets"]


def test_the_request_for_statistics_reaches_both_steps():
    preprocessor, _, vocabulary, data_sets = preprocessor_for()
    preprocessor.run(stats=True)

    assert vocabulary.stats is True
    assert data_sets.stats is True


def test_nothing_is_measured_by_default():
    preprocessor, _, vocabulary, data_sets = preprocessor_for()
    preprocessor.run()

    assert vocabulary.stats is False
    assert data_sets.stats is False
