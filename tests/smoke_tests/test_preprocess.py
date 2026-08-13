from pathlib import Path
from typing import Set

from silnlp.nmt.seq2seq_config import Seq2SeqConfig
from tests.smoke_tests.smoke_test_utils import (
    PREPROCESS_OUTPUT_PATTERNS,
    count_lines,
    delete_generated_paths,
    load_experiment_config,
    read_lines,
    run_preprocess_step,
    set_up_environment,
)

EXPERIMENT_NAME = "test_preprocess"

# The test and validation sizes that are configured in the experiment's config.yml
TEST_SIZE = 16
VAL_SIZE = 16


def test_preprocess_creates_data_sets():
    environment = set_up_environment()
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    delete_generated_paths(exp_dir, PREPROCESS_OUTPUT_PATTERNS)

    config = load_experiment_config(environment, EXPERIMENT_NAME, Seq2SeqConfig)
    run_preprocess_step(config, make_stats=True)

    check_tokenizer(exp_dir)
    check_train_data_set(exp_dir)
    check_val_data_set(exp_dir)
    check_test_data_set(exp_dir)
    check_data_sets_are_disjoint(exp_dir)
    check_tokenization_stats(exp_dir)

    delete_generated_paths(exp_dir, PREPROCESS_OUTPUT_PATTERNS)


def check_tokenizer(exp_dir: Path):
    # The experiment's tokenizer is saved to the experiment directory, so that the train, test and
    # translate steps all use the same one.
    for file_name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"]:
        assert (exp_dir / file_name).is_file()


def check_train_data_set(exp_dir: Path):
    train_line_count = count_lines(exp_dir / "train.src.txt")
    assert train_line_count > VAL_SIZE + TEST_SIZE
    assert count_lines(exp_dir / "train.trg.txt") == train_line_count
    assert count_lines(exp_dir / "train.vref.txt") == train_line_count

    # The detokenized files contain one line per verse and one line per term, while the tokenized
    # files contain an additional line for each term variant, so they are not the same length.
    detok_line_count = count_lines(exp_dir / "train.src.detok.txt")
    assert detok_line_count > 0
    assert count_lines(exp_dir / "train.trg.detok.txt") == detok_line_count
    assert detok_line_count <= train_line_count

    check_is_tokenized(exp_dir / "train.src.txt")
    check_is_tokenized(exp_dir / "train.trg.txt")
    check_is_detokenized(exp_dir / "train.src.detok.txt")
    check_is_detokenized(exp_dir / "train.trg.detok.txt")


def check_val_data_set(exp_dir: Path):
    for file_name in ["val.src.txt", "val.trg.txt", "val.vref.txt", "val.src.detok.txt", "val.trg.detok.txt"]:
        assert count_lines(exp_dir / file_name) == VAL_SIZE

    check_is_tokenized(exp_dir / "val.src.txt")
    check_is_detokenized(exp_dir / "val.src.detok.txt")


def check_test_data_set(exp_dir: Path):
    # The test step reads its source sentences from test.src.txt and its references from
    # test.trg.detok.txt, and identifies the verses through test.vref.txt.
    for file_name in ["test.src.txt", "test.src.detok.txt", "test.trg.detok.txt", "test.vref.txt"]:
        assert count_lines(exp_dir / file_name) == TEST_SIZE

    check_is_tokenized(exp_dir / "test.src.txt")
    check_is_detokenized(exp_dir / "test.src.detok.txt")
    check_is_detokenized(exp_dir / "test.trg.detok.txt")


def check_data_sets_are_disjoint(exp_dir: Path):
    train_vrefs = read_vrefs(exp_dir / "train.vref.txt")
    val_vrefs = read_vrefs(exp_dir / "val.vref.txt")
    test_vrefs = read_vrefs(exp_dir / "test.vref.txt")

    assert len(val_vrefs) == VAL_SIZE
    assert len(test_vrefs) == TEST_SIZE
    assert train_vrefs.isdisjoint(val_vrefs)
    assert train_vrefs.isdisjoint(test_vrefs)
    assert val_vrefs.isdisjoint(test_vrefs)


def check_tokenization_stats(exp_dir: Path):
    # These files are only created because the step is run with the --stats option.
    assert (exp_dir / "tokenization_stats.csv").is_file()
    assert (exp_dir / "tokenization_stats.xlsx").is_file()


def check_is_tokenized(path: Path):
    first_line = read_lines(path)[0]
    assert "▁" in first_line


def check_is_detokenized(path: Path):
    first_line = read_lines(path)[0]
    assert first_line != ""
    assert "▁" not in first_line


def read_vrefs(path: Path) -> Set[str]:
    # Rows that don't come from Scripture, e.g. the biblical terms in the training data, have an
    # empty verse reference.
    return {line for line in read_lines(path) if line != ""}
