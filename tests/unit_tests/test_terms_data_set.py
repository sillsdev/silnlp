from pathlib import Path
from typing import List

import pandas as pd
import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.terms_data_set import TermsDataSet
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


def terms(*rows, src_iso="en", trg_iso="es") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": [row[0] for row in rows],
            "target": [row[1] for row in rows],
            "vrefs": ["MAT 1:1"] * len(rows),
            "source_lang": [src_iso] * len(rows),
            "target_lang": [trg_iso] * len(rows),
        }
    )


def lines_of(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def test_an_empty_data_set_writes_nothing(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)

    assert data_set.write() == 0
    assert lines_of(files.train_source()) == []


def test_each_term_is_written_both_with_and_without_the_dummy_prefix(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)
    data_set.add(terms(("god", "dios")))

    assert data_set.write() == 2
    assert lines_of(files.train_source()) == ["_en|god", "en|god"]
    assert lines_of(files.train_target()) == ["_es|dios", "es|dios"]


def test_the_plain_term_is_written_once_however_many_variants_there_are(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)
    data_set.add(terms(("god", "dios")))
    data_set.write()

    assert lines_of(files.train_source_detokenized()) == ["god"]
    assert lines_of(files.train_target_detokenized()) == ["dios"]


def test_a_blank_verse_reference_is_written_for_every_variant(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)
    data_set.add(terms(("god", "dios")))
    data_set.write()

    assert lines_of(files.train_vref()) == ["", ""]


def test_each_term_is_tokenized_in_the_languages_of_its_own_corpus(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)
    data_set.add(terms(("god", "dios")))
    data_set.add(terms(("dieu", "gott"), src_iso="fr", trg_iso="de"))
    data_set.write()

    assert lines_of(files.train_source()) == ["_en|god", "en|god", "_fr|dieu", "fr|dieu"]
    assert lines_of(files.train_target()) == ["_es|dios", "es|dios", "_de|gott", "de|gott"]


def test_a_term_pair_repeated_across_corpora_is_written_once(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)
    data_set.add(terms(("god", "dios")))
    data_set.add(terms(("god", "dios")))

    assert data_set.write() == 2
    assert lines_of(files.train_source()) == ["_en|god", "en|god"]


def test_the_same_source_with_a_different_target_is_kept(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)
    data_set.add(terms(("god", "dios"), ("god", "senor")))

    assert data_set.write() == 4
    assert lines_of(files.train_target()) == ["_es|dios", "es|dios", "_es|senor", "es|senor"]


def test_tags_are_prefixed_to_the_source_only(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)
    data_set.add(terms(("god", "dios")), ["formal"])
    data_set.write()

    assert lines_of(files.train_source()) == ["_en|<formal> god", "en|<formal> god"]
    assert lines_of(files.train_target()) == ["_es|dios", "es|dios"]


def test_mirroring_writes_each_term_in_both_directions(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=True)
    data_set.add(terms(("god", "dios")))

    assert data_set.write() == 4
    assert lines_of(files.train_source()) == ["_es|dios", "es|dios", "_en|god", "en|god"]
    assert lines_of(files.train_target()) == ["_en|god", "en|god", "_es|dios", "es|dios"]


def test_mirroring_swaps_the_languages_along_with_the_terms(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=True)
    data_set.add(terms(("god", "dios")))
    data_set.write()

    assert lines_of(files.train_source_detokenized()) == ["dios", "god"]
    assert lines_of(files.train_target_detokenized()) == ["god", "dios"]


def test_mirroring_tags_the_reversed_direction_as_well(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=True)
    data_set.add(terms(("god", "dios")), ["formal"])
    data_set.write()

    assert lines_of(files.train_source()) == [
        "_es|<formal> dios",
        "es|<formal> dios",
        "_en|<formal> god",
        "en|<formal> god",
    ]


def test_mirroring_leaves_the_target_side_untagged_in_both_directions(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=True)
    data_set.add(terms(("god", "dios")), ["formal"])
    data_set.write()

    assert lines_of(files.train_target()) == ["_en|god", "en|god", "_es|dios", "es|dios"]


def test_a_corpus_with_no_terms_writes_nothing_but_is_still_accepted(files):
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)
    data_set.add(terms())

    assert data_set.write() == 0
    assert lines_of(files.train_source()) == []


def test_terms_are_appended_after_whatever_is_already_in_the_training_files(files):
    files.append(files.train_source(), ["_en|existing"])
    data_set = TermsDataSet(files, MarkingTokenizer(), mirror=False)
    data_set.add(terms(("god", "dios")))
    data_set.write()

    assert lines_of(files.train_source()) == ["_en|existing", "_en|god", "en|god"]
