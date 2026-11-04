import random
from pathlib import Path
from typing import List

import pytest

from silnlp.common.utils import NoiseMethod
from silnlp.nmt.alignment_scores import AlignmentScores
from silnlp.nmt.basic_data_set_writer import BasicDataSetWriter
from silnlp.nmt.corpora import DataFileType
from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles
from tests.unit_tests.conftest import MarkingTokenizer


class DropLastToken(NoiseMethod):
    def __call__(self, tokens: List[str]) -> List[str]:
        return tokens[:-1]


@pytest.fixture
def exp_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "exp"
    directory.mkdir()
    return directory


def writer_for(pairs, environment, exp_dir: Path, mirror: bool = False, multi_ref_eval: bool = False):
    inventory = CorpusInventory(pairs, include_glosses=False, environment=environment)
    files = ExperimentFiles(exp_dir, inventory, multi_ref_eval=multi_ref_eval)
    scores = AlignmentScores(exp_dir, "fast_align", force=False)
    return BasicDataSetWriter(files, inventory, MarkingTokenizer(), scores, mirror=mirror), files


def lines_of(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def test_a_training_pair_writes_every_sentence_to_the_training_files(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one", "two"])],
        [corpora.basic_file("es", "extra", ["uno", "dos"])],
        type=DataFileType.TRAIN,
    )
    writer, files = writer_for([pair], environment, exp_dir)

    assert writer.write(pair) == 2
    assert lines_of(files.train_source()) == ["_en|one", "_en|two"]
    assert lines_of(files.train_target()) == ["_es|uno", "_es|dos"]
    assert lines_of(files.validation_source()) == []


def test_a_sentence_pair_with_either_side_blank_is_skipped(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one", "", "three"])],
        [corpora.basic_file("es", "extra", ["uno", "dos", ""])],
        type=DataFileType.TRAIN,
    )
    writer, files = writer_for([pair], environment, exp_dir)

    assert writer.write(pair) == 1
    assert lines_of(files.train_source()) == ["_en|one"]


def test_tags_are_prefixed_to_the_source_only(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one"])],
        [corpora.basic_file("es", "extra", ["uno"])],
        type=DataFileType.TRAIN,
        tags=["formal"],
    )
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    assert lines_of(files.train_source()) == ["_en|<formal> one"]
    assert lines_of(files.train_target()) == ["_es|uno"]


def test_lexical_data_writes_a_second_variant_without_the_dummy_prefix(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one"])],
        [corpora.basic_file("es", "extra", ["uno"])],
        type=DataFileType.TRAIN | DataFileType.DICT,
        is_lexical_data=True,
    )
    writer, files = writer_for([pair], environment, exp_dir)

    assert writer.write(pair) == 2
    assert lines_of(files.train_source()) == ["_en|one", "en|one"]
    assert lines_of(files.train_target()) == ["_es|uno", "es|uno"]


def test_mirroring_also_writes_the_reverse_direction(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one"])],
        [corpora.basic_file("es", "extra", ["uno"])],
        type=DataFileType.TRAIN,
    )
    writer, files = writer_for([pair], environment, exp_dir, mirror=True)

    assert writer.write(pair) == 2
    assert lines_of(files.train_source()) == ["_en|one", "_es|uno"]
    assert lines_of(files.train_target()) == ["_es|uno", "_en|one"]


def test_noise_is_applied_to_the_training_source_only(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one two three"])],
        [corpora.basic_file("es", "extra", ["uno dos tres"])],
        type=DataFileType.TRAIN,
    )
    pair.src_noise = [DropLastToken()]
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    assert lines_of(files.train_source()) == ["_en|one two"]
    assert lines_of(files.train_target()) == ["_es|uno dos tres"]


def test_a_test_pair_normalizes_the_target_rather_than_tokenizing_it(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one"])],
        [corpora.basic_file("es", "extra", ["uno"])],
        type=DataFileType.TEST,
    )
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    assert lines_of(files.test_source("en", "es")) == ["_en|one"]
    assert lines_of(files.test_target("en", "es")) == ["norm(es|uno)"]


def test_a_validation_pair_tokenizes_both_sides(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one"])],
        [corpora.basic_file("es", "extra", ["uno"])],
        type=DataFileType.VAL,
    )
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    assert lines_of(files.validation_source()) == ["_en|one"]
    assert lines_of(files.validation_target()) == ["_es|uno"]


def test_the_configured_sizes_divide_the_corpus_between_the_data_sets(corpora, environment, exp_dir):
    random.seed(111)
    sentences = [f"line{index}" for index in range(6)]
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", sentences)],
        [corpora.basic_file("es", "extra", sentences)],
        test_size=1,
        val_size=2,
    )
    writer, files = writer_for([pair], environment, exp_dir)
    train_count = writer.write(pair)

    assert len(lines_of(files.test_source("en", "es"))) == 1
    assert len(lines_of(files.validation_source())) == 2
    assert train_count == 3
    assert len(lines_of(files.train_source())) == 3


def test_every_sentence_lands_in_exactly_one_data_set(corpora, environment, exp_dir):
    random.seed(7)
    sentences = [f"line{index}" for index in range(6)]
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", sentences)],
        [corpora.basic_file("es", "extra", sentences)],
        test_size=1,
        val_size=2,
    )
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    written = (
        lines_of(files.test_source("en", "es")) + lines_of(files.validation_source()) + lines_of(files.train_source())
    )
    assert sorted(written) == sorted(f"_en|{sentence}" for sentence in sentences)


def test_no_verse_references_are_written_without_scripture_data(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one"])],
        [corpora.basic_file("es", "extra", ["uno"])],
        type=DataFileType.TRAIN,
    )
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    assert lines_of(files.train_vref()) == []


def test_blank_verse_references_keep_the_files_aligned_when_the_experiment_has_scripture(corpora, environment, exp_dir):
    basic = corpora.pair(
        [corpora.basic_file("en", "extra", ["one"])],
        [corpora.basic_file("es", "extra", ["uno"])],
        type=DataFileType.TRAIN,
    )
    scripture = corpora.pair(
        [corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")], type=DataFileType.TRAIN
    )
    writer, files = writer_for([basic, scripture], environment, exp_dir)
    writer.write(basic)

    assert lines_of(files.train_vref()) == [""]


def test_a_dictionary_pair_writes_both_token_variants_of_every_sentence(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one"])],
        [corpora.basic_file("es", "extra", ["uno"])],
        type=DataFileType.DICT,
    )
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    assert lines_of(files.dictionary_source()) == ["_en|one\ten|one"]
    assert lines_of(files.dictionary_target()) == ["_es|uno\tes|uno"]
    assert lines_of(files.dictionary_vref()) == [""]


def test_every_file_pair_of_the_corpus_pair_is_written(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "first", ["one"]), corpora.basic_file("fr", "second", ["un"])],
        [corpora.basic_file("es", "first", ["uno"]), corpora.basic_file("de", "second", ["eins"])],
        type=DataFileType.TRAIN,
    )
    writer, files = writer_for([pair], environment, exp_dir)

    assert writer.write(pair) == 2
    assert lines_of(files.train_source()) == ["_en|one", "_fr|un"]
    assert lines_of(files.train_target()) == ["_es|uno", "_de|eins"]


class ScoresFromTable(AlignmentScores):
    """Alignment scores handed out by position, so no alignment tool has to run."""

    def __init__(self, exp_dir: Path, scores: List[float]) -> None:
        super().__init__(exp_dir, "fast_align", force=False)
        self._scores = scores

    def add_to(self, corpus, source_name: str, target_name: str) -> None:
        corpus["score"] = self._scores


def scored_writer_for(pairs, environment, exp_dir: Path, scores: List[float]):
    inventory = CorpusInventory(pairs, include_glosses=False, environment=environment)
    files = ExperimentFiles(exp_dir, inventory, multi_ref_eval=False)
    writer = BasicDataSetWriter(files, inventory, MarkingTokenizer(), ScoresFromTable(exp_dir, scores), mirror=False)
    return writer, files


def test_the_untokenized_sentences_are_written_beside_the_tokenized_ones(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one", "two", "three"])],
        [corpora.basic_file("es", "extra", ["uno", "dos", "tres"])],
        test_size=1,
        val_size=1,
    )
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    assert len(lines_of(files.train_source_detokenized())) == len(lines_of(files.train_source())) == 1
    assert not any("|" in line for line in lines_of(files.train_source_detokenized()))
    assert not any("|" in line for line in lines_of(files.validation_target_detokenized()))
    assert not any("|" in line for line in lines_of(files.test_source_detokenized("en", "es")))


def test_a_data_set_that_receives_no_sentences_leaves_no_file(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["one", "two"])],
        [corpora.basic_file("es", "extra", ["uno", "dos"])],
        type=DataFileType.TRAIN,
    )
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    assert files.train_source().is_file()
    assert not files.validation_source().exists()
    assert not files.test_source("en", "es").exists()


def test_a_file_excluded_from_the_test_set_contributes_nothing_to_it(corpora, environment, exp_dir):
    source = corpora.basic_file("en", "extra", ["one", "two", "three", "four"])
    source.include_test = False
    pair = corpora.pair(
        [source],
        [corpora.basic_file("es", "extra", ["uno", "dos", "tres", "cuatro"])],
        type=DataFileType.TRAIN | DataFileType.TEST,
        test_size=2,
    )
    writer, files = writer_for([pair], environment, exp_dir)
    writer.write(pair)

    assert not files.test_source("en", "es").exists()
    assert len(lines_of(files.train_source())) == 2


def test_the_training_share_is_taken_from_what_is_left_after_test_and_validation(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", [f"s{i}" for i in range(10)])],
        [corpora.basic_file("es", "extra", [f"t{i}" for i in range(10)])],
        size=0.5,
        test_size=2,
        val_size=2,
    )
    writer, files = writer_for([pair], environment, exp_dir)

    assert writer.write(pair) == 3
    assert len(lines_of(files.test_source("en", "es"))) == 2
    assert len(lines_of(files.validation_source())) == 2


def test_poorly_aligned_pairs_are_kept_out_of_training_and_validation(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["s0", "s1", "s2", "s3", "s4", "s5"])],
        [corpora.basic_file("es", "extra", ["t0", "t1", "t2", "t3", "t4", "t5"])],
        test_size=0,
        val_size=2,
        score_threshold=0.5,
    )
    writer, files = scored_writer_for([pair], environment, exp_dir, [0.9, 0.1, 0.9, 0.1, 0.9, 0.9])
    writer.write(pair)

    written = lines_of(files.train_source_detokenized()) + lines_of(files.validation_source_detokenized())
    assert sorted(written) == ["s0", "s2", "s4", "s5"]


def test_the_test_set_is_chosen_before_alignment_is_considered(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.basic_file("en", "extra", ["s0", "s1", "s2", "s3"])],
        [corpora.basic_file("es", "extra", ["t0", "t1", "t2", "t3"])],
        test_size=4,
        score_threshold=0.5,
    )
    writer, files = scored_writer_for([pair], environment, exp_dir, [0.1, 0.1, 0.1, 0.1])
    writer.write(pair)

    assert len(lines_of(files.test_source("en", "es"))) == 4
