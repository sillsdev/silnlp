import logging
from pathlib import Path

import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.shared_test_set import SharedTestSet


@pytest.fixture
def inventory(corpora, environment):
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    return CorpusInventory(pairs, include_glosses=False, environment=environment)


def write_vrefs(exp_dir: Path, file_name: str, vrefs) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / file_name).write_text("\n".join(vrefs) + "\n", encoding="utf-8")


def shared(environment, inventory, exp_name: str, current_exp_dir: Path) -> SharedTestSet:
    return SharedTestSet(exp_name, inventory, environment, current_exp_dir)


def test_verse_references_become_row_numbers_of_the_full_versification(environment, inventory, tmp_path):
    source_exp = environment.get_mt_exp_dir("source_experiment")
    write_vrefs(source_exp, "test.vref.txt", ["GEN 1:1", "GEN 1:3"])

    indices = shared(environment, inventory, "source_experiment", tmp_path / "current").indices_by_iso_pair()

    assert set(indices) == {("en", "es")}
    # GEN 1:1 is the first row of vref.txt and GEN 1:3 the third.
    assert indices[("en", "es")] == {0, 2}


def test_an_iso_qualified_file_keys_its_own_language_pair(environment, inventory, tmp_path):
    source_exp = environment.get_mt_exp_dir("source_experiment")
    write_vrefs(source_exp, "test.fr.de.vref.txt", ["GEN 1:1"])

    indices = shared(environment, inventory, "source_experiment", tmp_path / "current").indices_by_iso_pair()

    assert set(indices) == {("fr", "de")}


def test_several_test_files_are_all_read(environment, inventory, tmp_path):
    source_exp = environment.get_mt_exp_dir("source_experiment")
    write_vrefs(source_exp, "test.fr.de.vref.txt", ["GEN 1:1"])
    write_vrefs(source_exp, "test.en.es.vref.txt", ["GEN 1:2"])

    indices = shared(environment, inventory, "source_experiment", tmp_path / "current").indices_by_iso_pair()

    assert set(indices) == {("fr", "de"), ("en", "es")}


def test_a_verse_range_is_simplified_to_its_first_verse(environment, inventory, tmp_path):
    source_exp = environment.get_mt_exp_dir("source_experiment")
    write_vrefs(source_exp, "test.vref.txt", ["GEN 1:1-2"])

    indices = shared(environment, inventory, "source_experiment", tmp_path / "current").indices_by_iso_pair()

    assert indices[("en", "es")] == {0}


def test_blank_lines_in_the_test_file_are_skipped(environment, inventory, tmp_path):
    source_exp = environment.get_mt_exp_dir("source_experiment")
    (source_exp).mkdir(parents=True, exist_ok=True)
    (source_exp / "test.vref.txt").write_text("GEN 1:1\n\nGEN 1:3\n", encoding="utf-8")

    indices = shared(environment, inventory, "source_experiment", tmp_path / "current").indices_by_iso_pair()

    assert indices[("en", "es")] == {0, 2}


def test_an_experiment_with_no_test_files_warns(environment, inventory, tmp_path, caplog):
    environment.get_mt_exp_dir("source_experiment").mkdir(parents=True, exist_ok=True)
    current = tmp_path / "current"
    current.mkdir()

    with caplog.at_level(logging.WARNING):
        indices = shared(environment, inventory, "source_experiment", current).indices_by_iso_pair()

    assert indices == {}
    assert "does not contain any files" in caplog.text


def test_naming_an_experiment_that_does_not_exist_fails_instead_of_warning(environment, inventory, tmp_path):
    # The warning is never reached, because comparing against a missing directory raises first.
    current = tmp_path / "current"
    current.mkdir()

    with pytest.raises(FileNotFoundError):
        shared(environment, inventory, "no_such_experiment", current).indices_by_iso_pair()


def test_pointing_an_experiment_at_itself_warns_about_that_instead(environment, inventory, caplog):
    source_exp = environment.get_mt_exp_dir("source_experiment")
    source_exp.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.WARNING):
        shared(environment, inventory, "source_experiment", source_exp).indices_by_iso_pair()

    assert "same as the current experiment" in caplog.text
