from pathlib import Path
from typing import List, Optional, Set, Tuple

import pytest
from machine.scripture import get_books

from silnlp.nmt.corpora import DataFile
from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.terms import GlossLanguage, TermCategories
from silnlp.nmt.terms_data_set import TermsDataSet
from silnlp.nmt.terms_writer import TermsWriter
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


def writer_for(
    files: ExperimentFiles,
    environment,
    categories=None,
    gloss_language: Optional[GlossLanguage] = None,
    filter_books: Optional[Set[int]] = None,
    mirror: bool = False,
) -> TermsWriter:
    return TermsWriter(
        TermsDataSet(files, MarkingTokenizer(), mirror=mirror),
        TermCategories(categories),
        gloss_language if gloss_language is not None else GlossLanguage(False, set(), set()),
        filter_books,
        environment,
    )


def tagged(*data_files: DataFile, tags: List[str] = []) -> List[Tuple[DataFile, List[str]]]:
    return [(data_file, tags) for data_file in data_files]


def lines_of(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def test_a_term_in_both_lists_is_written_as_a_training_sentence(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("en", "BSB", ["god"])
    target = terms.renderings("es", "LBLA", ["dios"])

    assert writer_for(files, environment).write(tagged(source), tagged(target)) == 2
    assert lines_of(files.train_source()) == ["_en|god", "en|god"]
    assert lines_of(files.train_target()) == ["_es|dios", "es|dios"]


def test_a_term_the_target_list_has_no_rendering_for_is_skipped(corpora, environment, files):
    terms = corpora.terms_list().term("theos").term("kurios")
    source = terms.renderings("en", "BSB", ["god"], ["lord"])
    target = terms.renderings("es", "LBLA", ["dios"], [])

    writer_for(files, environment).write(tagged(source), tagged(target))

    assert lines_of(files.train_source()) == ["_en|god", "en|god"]


def test_every_combination_of_renderings_is_written(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("en", "BSB", ["god", "God"])
    target = terms.renderings("es", "LBLA", ["dios"])

    assert writer_for(files, environment).write(tagged(source), tagged(target)) == 4
    assert sorted(lines_of(files.train_source_detokenized())) == ["God", "god"]


def test_two_lists_in_the_same_language_are_not_paired(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("en", "BSB", ["god"])
    target = terms.renderings("en", "NIV", ["God"])

    assert writer_for(files, environment).write(tagged(source), tagged(target)) == 0
    assert lines_of(files.train_source()) == []


def test_every_source_list_is_paired_with_every_target_list(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    sources = [terms.renderings("en", "BSB", ["god"]), terms.renderings("fr", "LSG", ["dieu"])]
    targets = [terms.renderings("es", "LBLA", ["dios"]), terms.renderings("de", "LUT", ["gott"])]

    assert writer_for(files, environment).write(tagged(*sources), tagged(*targets)) == 8
    assert sorted(set(lines_of(files.train_source_detokenized()))) == ["dieu", "god"]
    assert sorted(set(lines_of(files.train_target_detokenized()))) == ["dios", "gott"]


def test_nothing_is_collected_when_every_category_is_excluded(corpora, environment, files):
    terms = corpora.terms_list().term("theos", category="PN")
    source = terms.renderings("en", "BSB", ["god"])
    target = terms.renderings("es", "LBLA", ["dios"])

    assert writer_for(files, environment, categories=[]).write(tagged(source), tagged(target)) == 0
    assert lines_of(files.train_source()) == []


def test_only_terms_in_the_configured_categories_are_collected(corpora, environment, files):
    terms = corpora.terms_list().term("theos", category="PN").term("bread", category="FL")
    source = terms.renderings("en", "BSB", ["god"], ["bread"])
    target = terms.renderings("es", "LBLA", ["dios"], ["pan"])

    writer_for(files, environment, categories=["PN"]).write(tagged(source), tagged(target))

    assert lines_of(files.train_source_detokenized()) == ["god"]


def test_a_term_outside_the_filtered_books_is_skipped(corpora, environment, files):
    terms = corpora.terms_list().term("theos", vrefs=["MAT 1:1"]).term("moses", vrefs=["GEN 1:1"])
    source = terms.renderings("en", "BSB", ["god"], ["moses"])
    target = terms.renderings("es", "LBLA", ["dios"], ["moises"])

    writer_for(files, environment, filter_books=get_books("MAT")).write(tagged(source), tagged(target))

    assert lines_of(files.train_source_detokenized()) == ["god"]


def test_glosses_become_targets_when_the_gloss_language_is_a_target_language(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("grc", "SBLGNT", ["theos"])
    target = terms.renderings("es", "LBLA", ["dios"])
    terms.glosses("en", ["god"])
    gloss_language = GlossLanguage(True, {"grc"}, {"en"})

    writer_for(files, environment, gloss_language=gloss_language).write(tagged(source), tagged(target))

    assert ("_grc|theos", "_en|god") in list(
        zip(lines_of(files.train_source()), lines_of(files.train_target()))
    )


def test_glosses_become_sources_when_the_gloss_language_is_a_source_language(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("en", "BSB", ["god"])
    target = terms.renderings("sw", "SWH", ["mungu"])
    terms.glosses("en", ["deity"])
    gloss_language = GlossLanguage(True, {"en"}, {"sw"})

    writer_for(files, environment, gloss_language=gloss_language).write(tagged(source), tagged(target))

    assert ("_en|deity", "_sw|mungu") in list(
        zip(lines_of(files.train_source()), lines_of(files.train_target()))
    )


def test_no_glosses_are_added_when_no_language_supports_them(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("grc", "SBLGNT", ["theos"])
    target = terms.renderings("sw", "SWH", ["mungu"])
    terms.glosses("en", ["god"])

    written = writer_for(files, environment, gloss_language=GlossLanguage(True, {"grc"}, {"sw"})).write(
        tagged(source), tagged(target)
    )

    assert written == 2
    assert lines_of(files.train_source_detokenized()) == ["theos"]


def test_the_tags_of_the_source_list_are_applied(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("en", "BSB", ["god"])
    target = terms.renderings("es", "LBLA", ["dios"])

    writer_for(files, environment).write(tagged(source, tags=["formal"]), tagged(target))

    assert lines_of(files.train_source()) == ["_en|<formal> god", "en|<formal> god"]


def test_mirroring_is_applied_to_the_collected_terms(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("en", "BSB", ["god"])
    target = terms.renderings("es", "LBLA", ["dios"])

    assert writer_for(files, environment, mirror=True).write(tagged(source), tagged(target)) == 4
    assert lines_of(files.train_source_detokenized()) == ["dios", "god"]
