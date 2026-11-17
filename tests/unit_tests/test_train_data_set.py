import random
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.train_data_set import TrainDataSet
from tests.unit_tests.conftest import MarkingTokenizer


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


def corpus(sources, targets, src_iso="en", trg_iso="es", vrefs=None, index=None) -> pd.DataFrame:
    # The index is the verse's position in the versification, which is what lines corpora up.
    return pd.DataFrame(
        {
            "vref": vrefs if vrefs is not None else [f"GEN 1:{i + 1}" for i in range(len(sources))],
            "source": sources,
            "target": targets,
            "source_lang": [src_iso] * len(sources),
            "target_lang": [trg_iso] * len(sources),
        },
        index=index if index is not None else range(len(sources)),
    )


def lines_of(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def test_one_corpus_is_written_with_its_languages(files):
    data_set = TrainDataSet(files, MarkingTokenizer(), mixed_source=False)
    data_set.add("BSB", "LBLA", [], corpus(["one", "two"], ["uno", "dos"]))

    assert data_set.write() == 2
    assert lines_of(files.train_source()) == ["_en|one", "_en|two"]
    assert lines_of(files.train_target()) == ["_es|uno", "_es|dos"]
    assert lines_of(files.train_vref()) == ["GEN 1:1", "GEN 1:2"]


def test_the_detokenized_text_is_written_alongside_the_tokenized(files):
    data_set = TrainDataSet(files, MarkingTokenizer(), mixed_source=False)
    data_set.add("BSB", "LBLA", [], corpus(["one"], ["uno"]))
    data_set.write()

    assert lines_of(files.train_source_detokenized()) == ["one"]
    assert lines_of(files.train_target_detokenized()) == ["uno"]


def test_several_corpora_are_appended_in_order(files):
    data_set = TrainDataSet(files, MarkingTokenizer(), mixed_source=False)
    data_set.add("BSB", "LBLA", [], corpus(["one"], ["uno"]))
    data_set.add("NIV", "RVR", [], corpus(["two"], ["dos"], src_iso="fr", trg_iso="de"))

    assert data_set.write() == 2
    assert lines_of(files.train_source()) == ["_en|one", "_fr|two"]
    assert lines_of(files.train_target()) == ["_es|uno", "_de|dos"]


def test_each_corpus_keeps_its_own_languages(files):
    data_set = TrainDataSet(files, MarkingTokenizer(), mixed_source=False)
    data_set.add("BSB", "LBLA", [], corpus(["one"], ["uno"], src_iso="en", trg_iso="es"))
    data_set.add("LSG", "LUT", [], corpus(["un"], ["eins"], src_iso="fr", trg_iso="de"))
    data_set.write()

    assert lines_of(files.train_source()) == ["_en|one", "_fr|un"]


def test_tags_are_prefixed_to_the_source(files):
    data_set = TrainDataSet(files, MarkingTokenizer(), mixed_source=False)
    data_set.add("BSB", "LBLA", ["formal"], corpus(["one"], ["uno"]))
    data_set.write()

    assert lines_of(files.train_source()) == ["_en|<formal> one"]
    assert lines_of(files.train_target()) == ["_es|uno"]


def test_an_empty_data_set_writes_nothing(files):
    data_set = TrainDataSet(files, MarkingTokenizer(), mixed_source=False)
    assert data_set.write() == 0
    assert lines_of(files.train_source()) == []


def test_mixed_source_keeps_one_column_per_source_project(files):
    data_set = TrainDataSet(files, MarkingTokenizer(), mixed_source=True)
    data_set.note_language("BSB", "en")
    data_set.note_language("NIV", "en")
    data_set.add("BSB", "LBLA", [], corpus(["from bsb"], ["uno"]))
    data_set.add("NIV", "LBLA", [], corpus(["from niv"], ["uno"]))

    assert data_set.write() == 1
    written = lines_of(files.train_source())
    assert written in (["_en|from bsb"], ["_en|from niv"])


def test_mixed_source_chooses_between_the_projects_that_have_a_sentence(files):
    # The verse is only in one of the two projects, so there is nothing to choose between.
    data_set = TrainDataSet(files, MarkingTokenizer(), mixed_source=True)
    data_set.note_language("BSB", "en")
    data_set.note_language("NIV", "fr")
    data_set.add("BSB", "LBLA", [], corpus(["only bsb"], ["uno"], vrefs=["GEN 1:1"], index=[0]))
    data_set.add("NIV", "LBLA", [], corpus(["only niv"], ["dos"], vrefs=["GEN 1:2"], index=[1]))

    data_set.write()

    assert sorted(lines_of(files.train_source())) == ["_en|only bsb", "_fr|only niv"]


def test_mixed_source_uses_the_language_of_the_project_it_chose(files):
    random.seed(3)
    data_set = TrainDataSet(files, MarkingTokenizer(), mixed_source=True)
    data_set.note_language("BSB", "en")
    data_set.note_language("LSG", "fr")
    data_set.add("BSB", "LBLA", [], corpus(["english"], ["uno"]))
    data_set.add("LSG", "LBLA", [], corpus(["french"], ["uno"]))
    data_set.write()

    written = lines_of(files.train_source())[0]
    assert written in ("_en|english", "_fr|french")
    # The marker language always matches the project the sentence came from.
    assert written.startswith("_en|") == written.endswith("english")
