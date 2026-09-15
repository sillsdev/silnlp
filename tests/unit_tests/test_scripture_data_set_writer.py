import random
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from silnlp.common.utils import NoiseMethod
from silnlp.nmt.alignment_scores import AlignmentScores, CorpusAligner
from silnlp.nmt.corpora import DataFileMapping, DataFileType
from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.scripture_data_set_writer import ScriptureDataSetWriter
from tests.unit_tests.conftest import MarkingTokenizer

GENESIS = 1


class DropLastToken(NoiseMethod):
    def __call__(self, tokens: List[str]) -> List[str]:
        return tokens[:-1]


class ScriptedAligner(CorpusAligner):
    """Scores a corpus without running a real aligner."""

    def __init__(self, scores: List[float]) -> None:
        self._scores = scores

    def add_scores(self, corpus: pd.DataFrame, aligner_id: str) -> None:
        corpus["score"] = self._scores[: len(corpus)]


@pytest.fixture
def exp_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "exp"
    directory.mkdir()
    return directory


def verses(count: int, prefix: str) -> List[str]:
    return [f"{prefix}{index}" for index in range(count)]


def writer_for(
    pairs,
    environment,
    exp_dir: Path,
    mirror: bool = False,
    multi_ref_eval: bool = False,
    aligner: CorpusAligner = None,
):
    inventory = CorpusInventory(pairs, include_glosses=False, environment=environment)
    files = ExperimentFiles(exp_dir, inventory, multi_ref_eval=multi_ref_eval)
    scores = AlignmentScores(
        exp_dir, "fast_align", force=False, aligner=aligner if aligner is not None else ScriptedAligner([])
    )
    return (
        lambda pair: ScriptureDataSetWriter(
            pair,
            files,
            inventory,
            MarkingTokenizer(),
            scores,
            environment,
            exp_dir,
            mirror=mirror,
            multi_ref_eval=multi_ref_eval,
        ).write(),
        files,
    )


def lines_of(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def test_a_training_pair_writes_every_verse_to_the_training_files(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(3, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(3, "trg"))],
        type=DataFileType.TRAIN,
    )
    write, files = writer_for([pair], environment, exp_dir)

    assert write(pair) == 3
    assert lines_of(files.train_source()) == ["_en|src0", "_en|src1", "_en|src2"]
    assert lines_of(files.train_target()) == ["_es|trg0", "_es|trg1", "_es|trg2"]


def test_the_verse_references_are_written_alongside_the_sentences(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(2, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(2, "trg"))],
        type=DataFileType.TRAIN,
    )
    write, files = writer_for([pair], environment, exp_dir)
    write(pair)

    assert lines_of(files.train_vref()) == ["GEN 1:1", "GEN 1:2"]


def test_a_verse_missing_from_either_side_is_left_out(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", ["src0", "", "src2"])],
        [corpora.scripture_file("es", "LBLA", ["trg0", "trg1", ""])],
        type=DataFileType.TRAIN,
    )
    write, files = writer_for([pair], environment, exp_dir)

    assert write(pair) == 1
    assert lines_of(files.train_source()) == ["_en|src0"]


def test_noise_is_applied_to_the_source_only(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", ["one two three"])],
        [corpora.scripture_file("es", "LBLA", ["uno dos tres"])],
        type=DataFileType.TRAIN,
        src_noise=[DropLastToken()],
    )
    write, files = writer_for([pair], environment, exp_dir)
    write(pair)

    assert lines_of(files.train_source()) == ["_en|one two"]
    assert lines_of(files.train_target()) == ["_es|uno dos tres"]


def test_only_the_named_corpus_books_are_used(corpora, environment, exp_dir):
    # Lines 1-31 are Genesis 1 and lines 32-56 are Genesis 2.
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(35, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(35, "trg"))],
        type=DataFileType.TRAIN,
        corpus_books={GENESIS: [2]},
    )
    write, files = writer_for([pair], environment, exp_dir)

    assert write(pair) == 4
    assert lines_of(files.train_vref()) == ["GEN 2:1", "GEN 2:2", "GEN 2:3", "GEN 2:4"]


def test_the_test_books_are_kept_out_of_the_training_set(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(35, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(35, "trg"))],
        type=DataFileType.TRAIN,
        test_books={GENESIS: [2]},
    )
    write, files = writer_for([pair], environment, exp_dir)

    assert write(pair) == 31
    assert lines_of(files.train_vref())[-1] == "GEN 1:31"


def test_the_named_test_books_become_the_test_set(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(35, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(35, "trg"))],
        type=DataFileType.TRAIN | DataFileType.TEST,
        test_books={GENESIS: [2]},
    )
    write, files = writer_for([pair], environment, exp_dir)
    write(pair)

    assert lines_of(files.test_vref("en", "es")) == ["GEN 2:1", "GEN 2:2", "GEN 2:3", "GEN 2:4"]


def test_the_test_set_is_split_out_of_the_corpus_when_no_test_books_are_named(corpora, environment, exp_dir):
    random.seed(11)
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(10, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(10, "trg"))],
        type=DataFileType.TRAIN | DataFileType.TEST,
        test_size=2,
    )
    write, files = writer_for([pair], environment, exp_dir)

    assert write(pair) == 8
    assert len(lines_of(files.test_source("en", "es"))) == 2


def test_the_validation_set_is_split_out_of_the_corpus(corpora, environment, exp_dir):
    random.seed(11)
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(10, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(10, "trg"))],
        type=DataFileType.TRAIN | DataFileType.VAL,
        val_size=3,
    )
    write, files = writer_for([pair], environment, exp_dir)

    assert write(pair) == 7
    assert len(lines_of(files.validation_source())) == 3


def test_the_test_and_validation_sets_take_different_verses(corpora, environment, exp_dir):
    random.seed(11)
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(10, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(10, "trg"))],
        type=DataFileType.TRAIN | DataFileType.TEST | DataFileType.VAL,
        test_size=3,
        val_size=3,
    )
    write, files = writer_for([pair], environment, exp_dir)
    write(pair)

    test = set(lines_of(files.test_vref("en", "es")))
    validation = set(lines_of(files.validation_vref()))
    assert test.isdisjoint(validation)
    assert len(test) == 3 and len(validation) == 3


def test_a_pair_that_is_not_for_testing_writes_no_test_set(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(4, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(4, "trg"))],
        type=DataFileType.TRAIN,
        test_size=2,
    )
    write, files = writer_for([pair], environment, exp_dir)

    assert write(pair) == 4
    assert lines_of(files.test_source("en", "es")) == []


def test_a_source_file_excluded_from_the_test_set_contributes_nothing_to_it(corpora, environment, exp_dir):
    random.seed(11)
    source = corpora.scripture_file("en", "BSB", verses(10, "src"))
    source.include_test = False
    pair = corpora.pair(
        [source],
        [corpora.scripture_file("es", "LBLA", verses(10, "trg"))],
        type=DataFileType.TRAIN | DataFileType.TEST,
        test_size=2,
    )
    write, files = writer_for([pair], environment, exp_dir)

    assert write(pair) == 8
    assert lines_of(files.test_source("en", "es")) == []


def test_mirroring_adds_the_reverse_direction_to_the_training_set(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(2, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(2, "trg"))],
        type=DataFileType.TRAIN,
    )
    write, files = writer_for([pair], environment, exp_dir, mirror=True)

    assert write(pair) == 4
    assert sorted(lines_of(files.train_source())) == ["_en|src0", "_en|src1", "_es|trg0", "_es|trg1"]


def test_every_file_pair_of_a_corpus_pair_tests_the_same_verses(corpora, environment, exp_dir):
    random.seed(11)
    pair = corpora.pair(
        [
            corpora.scripture_file("en", "BSB", verses(10, "src")),
            corpora.scripture_file("fr", "LSG", verses(10, "fra")),
        ],
        [
            corpora.scripture_file("es", "LBLA", verses(10, "trg")),
            corpora.scripture_file("de", "LUT", verses(10, "deu")),
        ],
        type=DataFileType.TRAIN | DataFileType.TEST,
        test_size=3,
    )
    write, files = writer_for([pair], environment, exp_dir)
    write(pair)

    assert lines_of(files.test_vref("en", "es")) == lines_of(files.test_vref("fr", "de"))


def test_verses_scoring_below_the_threshold_are_left_out_of_the_training_set(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(4, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(4, "trg"))],
        type=DataFileType.TRAIN,
        score_threshold=0.5,
    )
    write, files = writer_for([pair], environment, exp_dir, aligner=ScriptedAligner([0.9, 0.2, 0.0, 0.6]))

    assert write(pair) == 2
    assert lines_of(files.train_source()) == ["_en|src0", "_en|src3"]


def test_the_alignment_score_is_not_written_to_the_test_set(corpora, environment, exp_dir):
    random.seed(11)
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(6, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(6, "trg"))],
        type=DataFileType.TRAIN | DataFileType.TEST,
        test_size=2,
        score_threshold=0.1,
    )
    write, files = writer_for(
        [pair], environment, exp_dir, aligner=ScriptedAligner([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    )
    write(pair)

    assert len(lines_of(files.test_source("en", "es"))) == 2
    assert all("\t" not in line for line in lines_of(files.test_source("en", "es")))


def test_a_mixed_source_pair_chooses_one_source_project_per_verse(corpora, environment, exp_dir):
    random.seed(11)
    pair = corpora.pair(
        [
            corpora.scripture_file("en", "BSB", verses(4, "bsb")),
            corpora.scripture_file("en", "NIV", verses(4, "niv")),
        ],
        [corpora.scripture_file("es", "LBLA", verses(4, "trg"))],
        type=DataFileType.TRAIN,
        mapping=DataFileMapping.MIXED_SRC,
    )
    write, files = writer_for([pair], environment, exp_dir)

    assert write(pair) == 4
    written = lines_of(files.train_source())
    assert len(written) == 4
    assert all(line.startswith("_en|bsb") or line.startswith("_en|niv") for line in written)


def test_the_tags_of_the_pair_are_prefixed_to_the_training_source(corpora, environment, exp_dir):
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(1, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(1, "trg"))],
        type=DataFileType.TRAIN,
        tags=["formal"],
    )
    write, files = writer_for([pair], environment, exp_dir)
    write(pair)

    assert lines_of(files.train_source()) == ["_en|<formal> src0"]


def test_a_fractional_size_takes_that_share_of_the_training_corpus(corpora, environment, exp_dir):
    random.seed(11)
    pair = corpora.pair(
        [corpora.scripture_file("en", "BSB", verses(10, "src"))],
        [corpora.scripture_file("es", "LBLA", verses(10, "trg"))],
        type=DataFileType.TRAIN,
        size=0.5,
    )
    write, _ = writer_for([pair], environment, exp_dir)

    assert write(pair) == 5
