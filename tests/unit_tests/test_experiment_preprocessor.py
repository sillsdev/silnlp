from pathlib import Path
from typing import List

import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.experiment_preprocessor import ExperimentPreprocessor, TermsSettings


class RecordingWriter:
    """Stands in for one of the data set writers, recording what it was handed."""

    def __init__(self, count: int = 0) -> None:
        self._count = count
        self.calls: List = []

    def write(self, *args) -> int:
        self.calls.append(args)
        return self._count

    def writer_for(self, pair):
        self.calls.append(pair)
        return self

    def __call__(self, *args) -> int:
        return self.write(*args)


@pytest.fixture
def exp_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "exp"
    directory.mkdir()
    return directory


def terms(train: bool = False, dictionary: bool = False) -> TermsSettings:
    return TermsSettings({"train": train, "dictionary": dictionary})


def preprocessor_for(pairs, environment, exp_dir, scripture, basic, terms_writer, dictionary, settings, tokenize=True):
    inventory = CorpusInventory(pairs, include_glosses=False, environment=environment)
    files = ExperimentFiles(exp_dir, inventory, multi_ref_eval=False)
    return ExperimentPreprocessor(
        pairs, files, scripture, basic, terms_writer, dictionary, settings, tokenize=tokenize
    )


def test_a_scripture_pair_goes_to_the_scripture_writer(corpora, environment, exp_dir):
    pair = corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])
    scripture, basic = RecordingWriter(5), RecordingWriter(7)

    written = preprocessor_for(
        [pair], environment, exp_dir, scripture, basic, RecordingWriter(), RecordingWriter(), terms()
    ).write(stats=False)

    assert scripture.calls == [pair, ()]
    assert basic.calls == []
    assert written == 5


def test_a_basic_pair_goes_to_the_basic_writer(corpora, environment, exp_dir):
    pair = corpora.pair([corpora.basic_file("en", "extra")], [corpora.basic_file("es", "extra")])
    scripture, basic = RecordingWriter(5), RecordingWriter(7)

    written = preprocessor_for(
        [pair], environment, exp_dir, scripture, basic, RecordingWriter(), RecordingWriter(), terms()
    ).write(stats=False)

    assert basic.calls == [(pair,)]
    assert scripture.calls == []
    assert written == 7


def test_every_pair_contributes_to_the_training_count(corpora, environment, exp_dir):
    scripture_pair = corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])
    basic_pair = corpora.pair([corpora.basic_file("en", "extra")], [corpora.basic_file("es", "extra")])

    written = preprocessor_for(
        [scripture_pair, basic_pair], environment, exp_dir,
        RecordingWriter(5), RecordingWriter(7), RecordingWriter(), RecordingWriter(), terms(),
    ).write(stats=False)

    assert written == 12


def test_no_terms_are_written_when_the_config_asks_for_none(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB")],
        [corpora.scripture_file("es", "LBLA")],
        src_terms_files=[corpora.terms_file("en", "BSB")],
    )
    terms_writer, dictionary = RecordingWriter(3), RecordingWriter(4)

    preprocessor_for(
        [pair], environment, exp_dir, RecordingWriter(), RecordingWriter(), terms_writer, dictionary, terms()
    ).write(stats=False)

    assert terms_writer.calls == []
    assert dictionary.calls == []


def test_the_term_files_of_every_pair_reach_the_terms_writer(corpora, environment, exp_dir):
    source_terms = corpora.terms_file("en", "BSB")
    target_terms = corpora.terms_file("es", "LBLA")
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB")],
        [corpora.scripture_file("es", "LBLA")],
        tags=["formal"],
        src_terms_files=[source_terms],
        trg_terms_files=[target_terms],
    )
    terms_writer = RecordingWriter(3)

    written = preprocessor_for(
        [pair], environment, exp_dir, RecordingWriter(5), RecordingWriter(), terms_writer,
        RecordingWriter(), terms(train=True),
    ).write(stats=False)

    assert terms_writer.calls == [([(source_terms, ["formal"])], [(target_terms, ["formal"])])]
    assert written == 8


def test_the_dictionary_is_written_from_the_same_term_files(corpora, environment, exp_dir):
    source_terms = corpora.terms_file("en", "BSB")
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB")],
        [corpora.scripture_file("es", "LBLA")],
        src_terms_files=[source_terms],
    )
    dictionary = RecordingWriter(4)

    written = preprocessor_for(
        [pair], environment, exp_dir, RecordingWriter(5), RecordingWriter(), RecordingWriter(),
        dictionary, terms(dictionary=True),
    ).write(stats=False)

    assert dictionary.calls == [([(source_terms, [])], [])]
    # The dictionary is written alongside the training data rather than counted as part of it.
    assert written == 5


def test_the_previous_runs_data_sets_are_deleted_first(corpora, environment, exp_dir):
    pair = corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])
    inventory = CorpusInventory([pair], include_glosses=False, environment=environment)
    files = ExperimentFiles(exp_dir, inventory, multi_ref_eval=False)
    files.append(files.train_source(), ["left over"])

    ExperimentPreprocessor(
        [pair], files, RecordingWriter(), RecordingWriter(), RecordingWriter(), RecordingWriter(),
        terms(), tokenize=True,
    ).write(stats=False)

    assert not files.train_source().is_file()


class SentenceWritingWriter(RecordingWriter):
    """Writes a sentence to the training files, so the statistics have something to measure."""

    def __init__(self, files: ExperimentFiles) -> None:
        super().__init__(1)
        self._files = files

    def write(self, *args) -> int:
        super().write(*args)
        # Two sentences, because the report's standard deviation needs more than one data point.
        self._files.append(self._files.train_source(), ["a b c", "a b"])
        self._files.append(self._files.train_target(), ["d e f", "d e"])
        self._files.append(self._files.train_source_detokenized(), ["abc ab", "ab"])
        self._files.append(self._files.train_target_detokenized(), ["def de", "de"])
        return 1


def test_the_tokenization_report_is_written_when_statistics_are_asked_for(corpora, environment, exp_dir):
    pair = corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])
    inventory = CorpusInventory([pair], include_glosses=False, environment=environment)
    files = ExperimentFiles(exp_dir, inventory, multi_ref_eval=False)
    writer = SentenceWritingWriter(files)

    ExperimentPreprocessor(
        [pair], files, writer, RecordingWriter(), RecordingWriter(), RecordingWriter(),
        terms(), tokenize=True,
    ).write(stats=True)

    assert files.statistics_report().is_file()


def test_no_tokenization_report_is_written_when_the_corpora_are_not_tokenized(corpora, environment, exp_dir):
    pair = corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])
    inventory = CorpusInventory([pair], include_glosses=False, environment=environment)
    files = ExperimentFiles(exp_dir, inventory, multi_ref_eval=False)

    ExperimentPreprocessor(
        [pair], files, RecordingWriter(), RecordingWriter(), RecordingWriter(), RecordingWriter(),
        terms(), tokenize=False,
    ).write(stats=True)

    assert not files.statistics_report().is_file()


def test_term_files_are_gathered_for_either_use():
    assert TermsSettings({"train": True, "dictionary": False}).needs_term_files()
    assert TermsSettings({"train": False, "dictionary": True}).needs_term_files()
    assert not TermsSettings({"train": False, "dictionary": False}).needs_term_files()
