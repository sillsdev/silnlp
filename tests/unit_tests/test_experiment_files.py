from pathlib import Path

import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles


def files_for(pairs, environment, exp_dir: Path, multi_ref_eval: bool = False) -> ExperimentFiles:
    inventory = CorpusInventory(pairs, include_glosses=False, environment=environment)
    return ExperimentFiles(exp_dir, inventory, multi_ref_eval=multi_ref_eval)


@pytest.fixture
def one_iso_pair(corpora, environment, tmp_path):
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    return files_for(pairs, environment, tmp_path / "exp")


@pytest.fixture
def two_iso_pairs(corpora, environment, tmp_path):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("fr", "LSG")], [corpora.scripture_file("de", "LUT")]),
    ]
    return files_for(pairs, environment, tmp_path / "exp")


@pytest.fixture
def two_test_projects(corpora, environment, tmp_path):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "RVR")]),
    ]
    return files_for(pairs, environment, tmp_path / "exp")


def test_training_files_are_named_after_the_experiment_directory(one_iso_pair, tmp_path):
    exp_dir = tmp_path / "exp"
    assert one_iso_pair.train_source() == exp_dir / "train.src.txt"
    assert one_iso_pair.train_source_detokenized() == exp_dir / "train.src.detok.txt"
    assert one_iso_pair.train_target() == exp_dir / "train.trg.txt"
    assert one_iso_pair.train_target_detokenized() == exp_dir / "train.trg.detok.txt"
    assert one_iso_pair.train_vref() == exp_dir / "train.vref.txt"


def test_validation_source_files(one_iso_pair, tmp_path):
    exp_dir = tmp_path / "exp"
    assert one_iso_pair.validation_source() == exp_dir / "val.src.txt"
    assert one_iso_pair.validation_source_detokenized() == exp_dir / "val.src.detok.txt"
    assert one_iso_pair.validation_vref() == exp_dir / "val.vref.txt"


def test_a_single_validation_reference_is_unnumbered(one_iso_pair, tmp_path):
    exp_dir = tmp_path / "exp"
    assert one_iso_pair.validation_target() == exp_dir / "val.trg.txt"
    assert one_iso_pair.validation_target_detokenized() == exp_dir / "val.trg.detok.txt"


def test_multiple_validation_references_are_numbered(corpora, environment, tmp_path):
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    files = files_for(pairs, environment, tmp_path / "exp", multi_ref_eval=True)
    exp_dir = tmp_path / "exp"
    assert files.validation_target(0) == exp_dir / "val.trg.txt.0"
    assert files.validation_target(2) == exp_dir / "val.trg.txt.2"
    assert files.validation_target_detokenized(1) == exp_dir / "val.trg.detok.txt.1"


def test_dictionary_files(one_iso_pair, tmp_path):
    exp_dir = tmp_path / "exp"
    assert one_iso_pair.dictionary_source() == exp_dir / "dict.src.txt"
    assert one_iso_pair.dictionary_target() == exp_dir / "dict.trg.txt"
    assert one_iso_pair.dictionary_vref() == exp_dir / "dict.vref.txt"


def test_test_files_are_unqualified_for_a_single_iso_pair(one_iso_pair, tmp_path):
    exp_dir = tmp_path / "exp"
    assert one_iso_pair.test_source("en", "es") == exp_dir / "test.src.txt"
    assert one_iso_pair.test_source_detokenized("en", "es") == exp_dir / "test.src.detok.txt"
    assert one_iso_pair.test_vref("en", "es") == exp_dir / "test.vref.txt"


def test_test_files_carry_the_iso_pair_when_there_is_more_than_one(two_iso_pairs, tmp_path):
    exp_dir = tmp_path / "exp"
    assert two_iso_pairs.test_source("en", "es") == exp_dir / "test.en.es.src.txt"
    assert two_iso_pairs.test_source_detokenized("fr", "de") == exp_dir / "test.fr.de.src.detok.txt"
    assert two_iso_pairs.test_vref("fr", "de") == exp_dir / "test.fr.de.vref.txt"


def test_the_test_target_file_is_always_detokenized(one_iso_pair, tmp_path):
    assert one_iso_pair.test_target("en", "es") == tmp_path / "exp" / "test.trg.detok.txt"


def test_the_test_target_file_carries_the_project_when_there_is_more_than_one(two_test_projects, tmp_path):
    exp_dir = tmp_path / "exp"
    assert two_test_projects.test_target("en", "es", "LBLA") == exp_dir / "test.trg.detok.LBLA.txt"
    assert two_test_projects.test_target("en", "es", "RVR") == exp_dir / "test.trg.detok.RVR.txt"


def test_the_test_target_file_carries_both_the_iso_pair_and_the_project(corpora, environment, tmp_path):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "RVR")]),
        corpora.pair([corpora.scripture_file("fr", "LSG")], [corpora.scripture_file("de", "LUT")]),
    ]
    files = files_for(pairs, environment, tmp_path / "exp")
    assert files.test_target("en", "es", "LBLA") == tmp_path / "exp" / "test.en.es.trg.detok.LBLA.txt"


def test_appending_writes_to_the_named_file(one_iso_pair, tmp_path):
    (tmp_path / "exp").mkdir()
    one_iso_pair.append(one_iso_pair.train_source(), ["first", "second"])
    one_iso_pair.append(one_iso_pair.train_source(), ["third"])
    assert (tmp_path / "exp" / "train.src.txt").read_text(encoding="utf-8") == "first\nsecond\nthird\n"


def test_filling_writes_blank_lines(one_iso_pair, tmp_path):
    (tmp_path / "exp").mkdir()
    one_iso_pair.fill(one_iso_pair.train_target(), 3)
    assert (tmp_path / "exp" / "train.trg.txt").read_text(encoding="utf-8") == "\n\n\n"


def test_opening_for_append_keeps_what_is_already_there(one_iso_pair, tmp_path):
    (tmp_path / "exp").mkdir()
    one_iso_pair.append(one_iso_pair.train_vref(), ["GEN 1:1"])
    with one_iso_pair.open_for_append(one_iso_pair.train_vref()) as file:
        file.write("GEN 1:2\n")
    assert (tmp_path / "exp" / "train.vref.txt").read_text(encoding="utf-8") == "GEN 1:1\nGEN 1:2\n"


def test_deleting_the_data_sets_leaves_everything_else(one_iso_pair, tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    data_set_names = [
        "train.src.txt",
        "train.trg.detok.txt",
        "val.trg.txt.0",
        "val.src.txt",
        "test.en.es.src.txt",
        "dict.vref.txt",
    ]
    kept_names = ["config.yml", "tokenizer.json", "tokenization_stats.csv", "trainer_state.json"]
    for name in data_set_names + kept_names:
        (exp_dir / name).write_text("", encoding="utf-8")

    one_iso_pair.delete_data_sets()

    assert sorted(path.name for path in exp_dir.iterdir()) == sorted(kept_names + ["val.trg.txt.0"])
