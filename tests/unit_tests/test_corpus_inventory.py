from pathlib import Path

import pytest

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.corpora import DataFileType
from silnlp.nmt.corpus_inventory import CorpusInventory

TRAIN_TEST_VAL = DataFileType.TRAIN | DataFileType.TEST | DataFileType.VAL


def inventory(pairs, environment: SilNlpEnv, include_glosses=False) -> CorpusInventory:
    return CorpusInventory(pairs, include_glosses=include_glosses, environment=environment)


def test_an_experiment_with_no_corpus_pairs_is_empty(environment):
    empty = inventory([], environment)
    assert empty.source_isos() == set()
    assert empty.target_isos() == set()
    assert not empty.has_scripture_data()
    assert not empty.has_validation_split()
    assert not empty.spans_multiple_test_iso_pairs()
    assert empty.default_test_source_iso() == ""
    assert empty.default_test_target_iso() == ""


def test_isos_are_collected_from_every_pair(corpora, environment):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("fr", "LSG")], [corpora.scripture_file("de", "LUT")]),
    ]
    collected = inventory(pairs, environment)
    assert collected.source_isos() == {"en", "fr"}
    assert collected.target_isos() == {"es", "de"}


def test_only_test_pairs_contribute_test_isos(corpora, environment):
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")], type=DataFileType.TRAIN
        ),
        corpora.pair(
            [corpora.scripture_file("fr", "LSG")], [corpora.scripture_file("de", "LUT")], type=DataFileType.TEST
        ),
    ]
    collected = inventory(pairs, environment)
    assert collected.test_source_isos() == {"fr"}
    assert collected.test_target_isos() == {"de"}
    assert collected.source_isos() == {"en", "fr"}


def test_only_validation_pairs_contribute_validation_isos(corpora, environment):
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")], type=DataFileType.TRAIN
        ),
        corpora.pair(
            [corpora.scripture_file("fr", "LSG")], [corpora.scripture_file("de", "LUT")], type=DataFileType.VAL
        ),
    ]
    collected = inventory(pairs, environment)
    assert collected.default_validation_source_iso() == "fr"
    assert collected.default_validation_target_iso() == "de"


def test_the_default_test_isos_come_from_the_test_pairs(corpora, environment):
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    collected = inventory(pairs, environment)
    assert collected.default_test_source_iso() == "en"
    assert collected.default_test_target_iso() == "es"


def test_scripture_pairs_contribute_projects_but_basic_pairs_do_not(corpora, environment):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.basic_file("en", "extra")], [corpora.basic_file("es", "extra")]),
    ]
    collected = inventory(pairs, environment)
    assert collected.source_projects() == {"BSB"}
    assert collected.target_projects() == {"LBLA"}


def test_scripture_data_is_detected_from_any_pair(corpora, environment):
    basic_only = inventory(
        [corpora.pair([corpora.basic_file("en", "extra")], [corpora.basic_file("es", "extra")])], environment
    )
    assert not basic_only.has_scripture_data()

    mixed = inventory(
        [
            corpora.pair([corpora.basic_file("en", "extra")], [corpora.basic_file("es", "extra")]),
            corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        ],
        environment,
    )
    assert mixed.has_scripture_data()


def test_input_files_include_the_terms_files_of_scripture_pairs_only(corpora, environment):
    scripture_terms = corpora.terms_file("en", "BSB")
    basic_terms = corpora.terms_file("en", "OTHER")
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")],
            [corpora.scripture_file("es", "LBLA")],
            src_terms_files=[scripture_terms],
        ),
        corpora.pair(
            [corpora.basic_file("en", "extra")], [corpora.basic_file("es", "extra")], src_terms_files=[basic_terms]
        ),
    ]
    collected = inventory(pairs, environment)
    assert scripture_terms.path in collected.source_file_paths()
    assert basic_terms.path not in collected.source_file_paths()


def test_gloss_files_are_included_when_the_gloss_language_is_in_the_pair(corpora, environment):
    glosses = corpora.glosses_file("en")
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")],
            [corpora.scripture_file("es", "LBLA")],
            src_terms_files=[corpora.terms_file("en", "BSB")],
        )
    ]
    collected = inventory(pairs, environment, include_glosses=True)
    assert glosses in collected.source_file_paths()


def test_a_named_gloss_language_reaches_the_source_side_even_when_absent_from_the_pair(corpora, environment):
    glosses = corpora.glosses_file("fr")
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")],
            [corpora.scripture_file("de", "LUT")],
            src_terms_files=[corpora.terms_file("fr", "BSB")],
            trg_terms_files=[corpora.terms_file("fr", "LUT")],
        )
    ]
    collected = inventory(pairs, environment, include_glosses="fr")
    assert glosses in collected.source_file_paths()
    assert glosses not in collected.target_file_paths()


def test_gloss_files_reach_the_target_side_when_its_language_is_a_gloss_language(corpora, environment):
    glosses = corpora.glosses_file("en")
    pairs = [
        corpora.pair(
            [corpora.scripture_file("de", "LUT")],
            [corpora.scripture_file("en", "BSB")],
            trg_terms_files=[corpora.terms_file("en", "BSB")],
        )
    ]
    collected = inventory(pairs, environment, include_glosses=True)
    assert glosses in collected.target_file_paths()
    assert glosses not in collected.source_file_paths()


def test_no_gloss_files_are_included_when_glosses_are_off(corpora, environment):
    glosses = corpora.glosses_file("en")
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")],
            [corpora.scripture_file("es", "LBLA")],
            src_terms_files=[corpora.terms_file("en", "BSB")],
        )
    ]
    collected = inventory(pairs, environment, include_glosses=False)
    assert glosses not in collected.source_file_paths()


def test_tags_are_wrapped_in_angle_brackets(corpora, environment):
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")], tags=["formal", "ot"]
        )
    ]
    assert inventory(pairs, environment).tags() == {"<formal>", "<ot>"}


def test_missing_input_files_are_reported(corpora, environment):
    source = corpora.scripture_file("en", "BSB")
    target = corpora.scripture_file("es", "LBLA")
    collected = inventory([corpora.pair([source], [target])], environment)
    assert collected.missing_input_files() == []

    target.path.unlink()
    assert collected.missing_input_files() == [target.path]


def test_a_validation_split_needs_a_validation_pair_with_a_size(corpora, environment):
    source, target = corpora.scripture_file("en", "BSB"), corpora.scripture_file("es", "LBLA")
    assert inventory([corpora.pair([source], [target], val_size=8)], environment).has_validation_split()
    assert not inventory([corpora.pair([source], [target], val_size=0)], environment).has_validation_split()
    assert not inventory(
        [corpora.pair([source], [target], type=DataFileType.TRAIN, val_size=8)], environment
    ).has_validation_split()


def test_a_validation_pair_without_its_own_size_falls_back_to_the_pair_size(corpora, environment):
    source, target = corpora.scripture_file("en", "BSB"), corpora.scripture_file("es", "LBLA")
    assert inventory([corpora.pair([source], [target], size=1.0, val_size=None)], environment).has_validation_split()
    assert not inventory([corpora.pair([source], [target], size=0, val_size=None)], environment).has_validation_split()


def test_test_projects_come_from_the_target_side_of_scripture_test_pairs(corpora, environment):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "RVR")]),
    ]
    collected = inventory(pairs, environment)
    assert collected.test_projects("en", "es") == {"LBLA", "RVR"}
    assert collected.has_multiple_test_projects("en", "es")


def test_a_single_test_project_is_not_multiple(corpora, environment):
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    collected = inventory(pairs, environment)
    assert collected.test_projects("en", "es") == {"LBLA"}
    assert not collected.has_multiple_test_projects("en", "es")


def test_basic_test_data_is_reported_under_its_own_project_name(corpora, environment):
    pairs = [corpora.pair([corpora.basic_file("en", "extra")], [corpora.basic_file("es", "extra")])]
    collected = inventory(pairs, environment)
    assert collected.test_projects("en", "es") == {"BASIC"}


def test_validation_projects_are_counted_only_for_scripture_pairs(corpora, environment):
    scripture = inventory(
        [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])], environment
    )
    assert scripture.validation_project_count("en", "es") == 1

    basic = inventory(
        [corpora.pair([corpora.basic_file("en", "extra")], [corpora.basic_file("es", "extra")])], environment
    )
    assert basic.validation_project_count("en", "es") == 0


def test_a_single_test_iso_pair_does_not_span_multiple(corpora, environment):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("en", "NIV")], [corpora.scripture_file("es", "RVR")]),
    ]
    assert not inventory(pairs, environment).spans_multiple_test_iso_pairs()


def test_two_test_iso_pairs_span_multiple(corpora, environment):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair([corpora.scripture_file("fr", "LSG")], [corpora.scripture_file("de", "LUT")]),
    ]
    assert inventory(pairs, environment).spans_multiple_test_iso_pairs()


def test_iso_pairs_without_test_data_do_not_count_towards_spanning(corpora, environment):
    pairs = [
        corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")]),
        corpora.pair(
            [corpora.scripture_file("fr", "LSG")], [corpora.scripture_file("de", "LUT")], type=DataFileType.TRAIN
        ),
    ]
    assert not inventory(pairs, environment).spans_multiple_test_iso_pairs()


def test_a_project_is_a_train_project_when_a_training_pair_uses_it(corpora, environment):
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")], type=DataFileType.TRAIN
        ),
        corpora.pair(
            [corpora.scripture_file("en", "NIV")], [corpora.scripture_file("es", "RVR")], type=DataFileType.TEST
        ),
    ]
    collected = inventory(pairs, environment)
    assert collected.is_train_project("es", "LBLA")
    assert not collected.is_train_project("es", "RVR")


def test_a_train_project_is_matched_on_both_iso_and_project(corpora, environment):
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")], type=DataFileType.TRAIN
        )
    ]
    collected = inventory(pairs, environment)
    assert not collected.is_train_project("fr", "LBLA")
    assert not collected.is_train_project("es", "RVR")


def test_a_prediction_file_names_the_project_it_is_a_reference_for(corpora, environment):
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    collected = inventory(pairs, environment)
    assert collected.references_one_of({"BSB"}, Path("test.trg-predictions.detok.BSB.txt"))
    assert not collected.references_one_of({"LBLA"}, Path("test.trg-predictions.detok.BSB.txt"))


def test_a_prediction_file_qualified_by_its_iso_pair_names_its_project(corpora, environment):
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    collected = inventory(pairs, environment)
    assert collected.references_one_of({"LBLA"}, Path("test.en.es.trg-predictions.detok.LBLA.txt"))
    assert not collected.references_one_of({"BSB"}, Path("test.en.es.trg-predictions.detok.LBLA.txt"))


def test_a_prediction_file_is_recognised_as_a_training_reference(corpora, environment):
    pairs = [
        corpora.pair(
            [corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")], type=DataFileType.TRAIN
        ),
        corpora.pair(
            [corpora.scripture_file("en", "NIV")], [corpora.scripture_file("es", "RVR")], type=DataFileType.TEST
        ),
    ]
    collected = inventory(pairs, environment)
    assert collected.is_train_reference(Path("test.en.es.trg-predictions.detok.LBLA.txt"))
    assert not collected.is_train_reference(Path("test.en.es.trg-predictions.detok.RVR.txt"))
