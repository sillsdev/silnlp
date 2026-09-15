from pathlib import Path
from typing import List

import pandas as pd
import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.eval_data_set import EvalDataSet
from silnlp.nmt.eval_data_set_writer import ScriptureTestSetWriter, ScriptureValidationSetWriter
from silnlp.nmt.experiment_files import ExperimentFiles
from tests.unit_tests.conftest import MarkingTokenizer


@pytest.fixture
def exp_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "exp"
    directory.mkdir()
    return directory


def corpus(sources, targets, index=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vref": [f"GEN 1:{i + 1}" for i in range(len(sources))],
            "source": sources,
            "target": targets,
        },
        index=index if index is not None else range(len(sources)),
    )


def lines_of(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def files_for(corpora, environment, exp_dir, target_projects=("LBLA",), multi_ref_eval=False):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", project)])
        for project in target_projects
    ]
    inventory = CorpusInventory(pairs, include_glosses=False, environment=environment)
    return ExperimentFiles(exp_dir, inventory, multi_ref_eval=multi_ref_eval), inventory


def test_the_validation_source_is_written_tokenized_and_plain(corpora, environment, exp_dir):
    files, _ = files_for(corpora, environment, exp_dir)
    validation = EvalDataSet()
    validation.add("en", "es", "LBLA", [], corpus(["one", "two"], ["uno", "dos"]))

    assert ScriptureValidationSetWriter(files, MarkingTokenizer(), multi_ref_eval=False).write(validation) == 2
    assert lines_of(files.validation_source()) == ["_en|one", "_en|two"]
    assert lines_of(files.validation_source_detokenized()) == ["one", "two"]
    assert lines_of(files.validation_vref()) == ["GEN 1:1", "GEN 1:2"]


def test_a_single_reference_is_written_tokenized_and_plain(corpora, environment, exp_dir):
    files, _ = files_for(corpora, environment, exp_dir)
    validation = EvalDataSet()
    validation.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))

    ScriptureValidationSetWriter(files, MarkingTokenizer(), multi_ref_eval=False).write(validation)

    assert lines_of(files.validation_target()) == ["_es|uno"]
    assert lines_of(files.validation_target_detokenized()) == ["uno"]


def test_multiple_references_are_written_to_numbered_files(corpora, environment, exp_dir):
    files, _ = files_for(corpora, environment, exp_dir, target_projects=("LBLA", "RVR"), multi_ref_eval=True)
    validation = EvalDataSet()
    validation.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))
    validation.add("en", "es", "RVR", [], corpus(["one"], ["uno rvr"]))

    ScriptureValidationSetWriter(files, MarkingTokenizer(), multi_ref_eval=True).write(validation)

    assert lines_of(files.validation_target(0))[0] == "_es|uno"
    assert lines_of(files.validation_target(1))[0] == "_es|uno rvr"


def test_multiple_references_write_the_plain_text_to_the_same_numbered_files(corpora, environment, exp_dir):
    # Only the single-reference path has a separate detokenized file, so with several references
    # the plain text is appended to the numbered files after the tokenized text.
    files, _ = files_for(corpora, environment, exp_dir, target_projects=("LBLA", "RVR"), multi_ref_eval=True)
    validation = EvalDataSet()
    validation.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))
    validation.add("en", "es", "RVR", [], corpus(["one"], ["uno rvr"]))

    ScriptureValidationSetWriter(files, MarkingTokenizer(), multi_ref_eval=True).write(validation)

    assert lines_of(files.validation_target(0)) == ["_es|uno", "uno"]
    assert lines_of(files.validation_target_detokenized(0)) == []


def test_a_reference_that_has_no_verse_is_left_blank(corpora, environment, exp_dir):
    files, _ = files_for(corpora, environment, exp_dir, target_projects=("LBLA", "RVR"), multi_ref_eval=True)
    validation = EvalDataSet()
    validation.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))

    ScriptureValidationSetWriter(files, MarkingTokenizer(), multi_ref_eval=True).write(validation)

    assert lines_of(files.validation_target(0))[0] == "_es|uno"
    assert lines_of(files.validation_target(1))[0] == ""


def test_one_reference_is_chosen_per_verse_when_only_one_is_kept(corpora, environment, exp_dir):
    files, _ = files_for(corpora, environment, exp_dir, target_projects=("LBLA", "RVR"))
    validation = EvalDataSet()
    validation.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))
    validation.add("en", "es", "RVR", [], corpus(["one"], ["uno rvr"]))

    ScriptureValidationSetWriter(files, MarkingTokenizer(), multi_ref_eval=False).write(validation)

    assert lines_of(files.validation_target())[0] in ("_es|uno", "_es|uno rvr")
    assert len(lines_of(files.validation_target())) == 1


def test_the_test_source_is_written_with_its_verse_references(corpora, environment, exp_dir):
    files, inventory = files_for(corpora, environment, exp_dir)
    test = EvalDataSet()
    test.add("en", "es", "LBLA", [], corpus(["one", "two"], ["uno", "dos"]))

    assert ScriptureTestSetWriter(files, inventory, MarkingTokenizer()).write(test) == 2
    assert lines_of(files.test_vref("en", "es")) == ["GEN 1:1", "GEN 1:2"]
    assert lines_of(files.test_source("en", "es")) == ["_en|one", "_en|two"]
    assert lines_of(files.test_source_detokenized("en", "es")) == ["one", "two"]


def test_the_test_target_is_normalized_rather_than_tokenized(corpora, environment, exp_dir):
    files, inventory = files_for(corpora, environment, exp_dir)
    test = EvalDataSet()
    test.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))

    ScriptureTestSetWriter(files, inventory, MarkingTokenizer()).write(test)

    assert lines_of(files.test_target("en", "es", "LBLA")) == ["norm(es|uno)"]


def test_each_test_project_gets_its_own_target_file(corpora, environment, exp_dir):
    files, inventory = files_for(corpora, environment, exp_dir, target_projects=("LBLA", "RVR"))
    test = EvalDataSet()
    test.add("en", "es", "LBLA", [], corpus(["one"], ["uno"]))
    test.add("en", "es", "RVR", [], corpus(["one"], ["uno rvr"]))

    ScriptureTestSetWriter(files, inventory, MarkingTokenizer()).write(test)

    assert lines_of(files.test_target("en", "es", "LBLA")) == ["norm(es|uno)"]
    assert lines_of(files.test_target("en", "es", "RVR")) == ["norm(es|uno rvr)"]


def test_a_test_project_that_contributed_nothing_is_padded_to_the_same_length(corpora, environment, exp_dir):
    files, inventory = files_for(corpora, environment, exp_dir, target_projects=("LBLA", "RVR"))
    test = EvalDataSet()
    test.add("en", "es", "LBLA", [], corpus(["one", "two"], ["uno", "dos"]))

    ScriptureTestSetWriter(files, inventory, MarkingTokenizer()).write(test)

    assert len(lines_of(files.test_target("en", "es", "LBLA"))) == 2
    assert lines_of(files.test_target("en", "es", "RVR")) == ["", ""]
