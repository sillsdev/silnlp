from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from silnlp.nmt.corpora import DataFile
from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.dictionary_writer import NoDictionaryWriter, TermDictionaryWriter
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.terms import GlossLanguage, TermCategories
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
) -> TermDictionaryWriter:
    return TermDictionaryWriter(
        files,
        MarkingTokenizer(),
        TermCategories(categories),
        gloss_language if gloss_language is not None else GlossLanguage(False, set(), set()),
        environment,
    )


def tagged(*data_files: DataFile) -> List[Tuple[DataFile, List[str]]]:
    return [(data_file, []) for data_file in data_files]


def lines_of(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def test_every_rendering_is_written_with_and_without_the_dummy_prefix(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    target = terms.renderings("es", "LBLA", ["dios", "senor"])

    assert writer_for(files, environment).write([], tagged(target)) == 1
    assert lines_of(files.dictionary_target()) == ["_es*|dios\tes*|dios\t_es*|senor\tes*|senor"]


def test_the_verse_references_of_the_term_are_written_alongside_it(corpora, environment, files):
    terms = corpora.terms_list().term("theos", vrefs=["MAT 1:1", "MAT 1:2"])
    target = terms.renderings("es", "LBLA", ["dios"])

    writer_for(files, environment).write([], tagged(target))

    assert lines_of(files.dictionary_vref())[0].split("\t") == sorted(["MAT 1:1", "MAT 1:2"])


def test_a_term_with_no_renderings_is_left_out(corpora, environment, files):
    terms = corpora.terms_list().term("theos").term("kurios")
    target = terms.renderings("es", "LBLA", ["dios"], [])

    assert writer_for(files, environment).write([], tagged(target)) == 1
    assert lines_of(files.dictionary_target()) == ["_es*|dios\tes*|dios"]


def test_terms_outside_the_configured_categories_are_left_out(corpora, environment, files):
    terms = corpora.terms_list().term("theos", category="PN").term("bread", category="FL")
    target = terms.renderings("es", "LBLA", ["dios"], ["pan"])

    assert writer_for(files, environment, categories=["PN"]).write([], tagged(target)) == 1
    assert lines_of(files.dictionary_target()) == ["_es*|dios\tes*|dios"]


def test_nothing_is_written_when_every_category_is_excluded(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    target = terms.renderings("es", "LBLA", ["dios"])

    assert writer_for(files, environment, categories=[]).write([], tagged(target)) == 0
    assert lines_of(files.dictionary_target()) == []


def test_the_dictionary_files_are_created_even_when_nothing_is_written(corpora, environment, files):
    # Downstream steps read these files unconditionally, so they have to exist.
    terms = corpora.terms_list().term("theos")
    target = terms.renderings("es", "LBLA", ["dios"])

    writer_for(files, environment, categories=[]).write([], tagged(target))

    assert files.dictionary_target().is_file()
    assert files.dictionary_vref().is_file()


def test_each_target_list_is_tokenized_in_its_own_language(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    spanish = terms.renderings("es", "LBLA", ["dios"])
    german = terms.renderings("de", "LUT", ["gott"])

    assert writer_for(files, environment).write([], tagged(spanish, german)) == 2
    assert lines_of(files.dictionary_target()) == ["_es*|dios\tes*|dios", "_de*|gott\tde*|gott"]


def test_the_glosses_of_the_source_terms_are_added_when_a_gloss_language_is_available(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("grc", "SBLGNT", ["theos"])
    target = terms.renderings("es", "LBLA", ["dios"])
    terms.glosses("en", ["god", "deity"])
    gloss_language = GlossLanguage(True, {"grc"}, {"en"})

    assert writer_for(files, environment, gloss_language=gloss_language).write(tagged(source), tagged(target)) == 2
    assert lines_of(files.dictionary_target())[1] == "_en*|god\ten*|god\t_en*|deity\ten*|deity"


def test_no_glosses_are_added_when_no_language_supports_them(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    source = terms.renderings("grc", "SBLGNT", ["theos"])
    target = terms.renderings("sw", "SWH", ["mungu"])
    terms.glosses("en", ["god"])
    gloss_language = GlossLanguage(True, {"grc"}, {"sw"})

    assert writer_for(files, environment, gloss_language=gloss_language).write(tagged(source), tagged(target)) == 1
    assert lines_of(files.dictionary_target()) == ["_sw*|mungu\tsw*|mungu"]


def test_a_source_term_with_no_glosses_is_left_out(corpora, environment, files):
    terms = corpora.terms_list().term("theos").term("kurios")
    source = terms.renderings("grc", "SBLGNT", ["theos"], ["kurios"])
    target = terms.renderings("es", "LBLA", ["dios"], ["senor"])
    terms.glosses("en", ["god"], [])
    gloss_language = GlossLanguage(True, {"grc"}, {"en"})

    assert writer_for(files, environment, gloss_language=gloss_language).write(tagged(source), tagged(target)) == 3


def test_a_model_that_takes_no_dictionary_writes_nothing(corpora, environment, files):
    terms = corpora.terms_list().term("theos")
    target = terms.renderings("es", "LBLA", ["dios"])

    assert NoDictionaryWriter().write([], tagged(target)) == 0
    assert lines_of(files.dictionary_target()) == []
