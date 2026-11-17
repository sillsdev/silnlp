from pathlib import Path
from typing import List

import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.training_data_sets import Seq2SeqTrainingDataSets, TokenizedBatchEncoder


class StandInTokenizer:
    """Encodes each token as its length, which makes the encoded ids readable in the assertions."""

    pad_token_id = 0

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [len(token) for token in tokens]

    def pad(self, encoded: dict, padding: bool, return_tensors) -> dict:
        return {"input_ids": encoded["input_ids"]}


class StandInPretrainedTokenizer:
    def __init__(self) -> None:
        self._tokenizer = StandInTokenizer()

    def load(self) -> StandInTokenizer:
        return self._tokenizer


class StandInTrainingArguments:
    """Stands in for the training arguments, whose only part used here is the process guard."""

    class _Guard:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def main_process_first(self, desc: str):
        return self._Guard()


@pytest.fixture
def exp_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "exp"
    directory.mkdir()
    return directory


@pytest.fixture
def files(corpora, environment, exp_dir) -> ExperimentFiles:
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    inventory = CorpusInventory(pairs, include_glosses=False, environment=environment)
    return ExperimentFiles(exp_dir, inventory, multi_ref_eval=False)


def data_sets_for(files: ExperimentFiles) -> Seq2SeqTrainingDataSets:
    return Seq2SeqTrainingDataSets(files, StandInPretrainedTokenizer())


def write(path: Path, lines: List[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def test_nothing_is_read_when_the_training_files_are_absent(files):
    assert data_sets_for(files).training(StandInTrainingArguments()) is None


def test_nothing_is_read_when_only_one_side_is_present(files):
    write(files.train_source(), ["one"])

    assert data_sets_for(files).training(StandInTrainingArguments()) is None


def test_each_sentence_pair_becomes_one_encoded_row(files):
    write(files.train_source(), ["a bb", "ccc"])
    write(files.train_target(), ["dd", "e ffff"])

    data_set = data_sets_for(files).training(StandInTrainingArguments())

    assert len(data_set) == 2


def test_the_source_tokens_become_the_model_input(files):
    write(files.train_source(), ["a bb ccc"])
    write(files.train_target(), ["d"])

    data_set = data_sets_for(files).training(StandInTrainingArguments())

    assert data_set[0]["input_ids"] == [1, 2, 3]


def test_the_target_tokens_become_the_labels(files):
    write(files.train_source(), ["a"])
    write(files.train_target(), ["dd eeee"])

    data_set = data_sets_for(files).training(StandInTrainingArguments())

    assert data_set[0]["labels"] == [2, 4]


def test_surrounding_whitespace_is_stripped_from_each_line(files):
    write(files.train_source(), ["  a bb  "])
    write(files.train_target(), ["  ccc  "])

    data_set = data_sets_for(files).training(StandInTrainingArguments())

    assert data_set[0]["input_ids"] == [1, 2]


def test_the_validation_set_is_read_from_the_validation_files(files):
    write(files.validation_source(), ["aaa"])
    write(files.validation_target(), ["b"])

    data_set = data_sets_for(files).validation(StandInTrainingArguments())

    assert len(data_set) == 1
    assert data_set[0]["input_ids"] == [3]


def test_the_validation_set_is_absent_when_its_files_are(files):
    write(files.train_source(), ["a"])
    write(files.train_target(), ["b"])

    assert data_sets_for(files).validation(StandInTrainingArguments()) is None


def test_already_tokenized_sentences_are_encoded_as_they_are():
    encoded = TokenizedBatchEncoder(StandInTokenizer()).encode([["a", "bb"], ["ccc"]])

    assert encoded["input_ids"] == [[1, 2], [3]]
