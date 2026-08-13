import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import pytest

from silnlp.nmt import test as nmt_test
from silnlp.nmt.seq2seq_config import Seq2SeqConfig
from tests.smoke_tests.mock_pretrained_model import (
    FixedTranslationPreTrainedModelProviderFactory,
    mock_sequence_log_prob,
)
from tests.smoke_tests.smoke_test_utils import (
    TEST_OUTPUT_PATTERNS,
    create_model_with_mock_pretrained_model,
    delete_generated_paths,
    load_experiment_config,
    read_lines,
    set_up_environment,
)

EXPERIMENT_NAME = "test_test"

# The number of test sentences in the test data that is stored in the experiment directory, and
# the step of the checkpoint directory that is stored there. The test step resolves the last
# checkpoint from the directory names under run/, so the step is also the suffix of every file
# that the test step creates.
TEST_SIZE = 8
CHECKPOINT_STEP = 2


def test_test_scores_the_last_checkpoint():
    environment = set_up_environment()
    exp_dir = environment.get_mt_exp_dir(EXPERIMENT_NAME)
    delete_generated_paths(exp_dir, TEST_OUTPUT_PATTERNS)

    # Inferencing is done by a mock model that "translates" every sentence to the same fixed
    # sentence.
    config = load_experiment_config(environment, EXPERIMENT_NAME, Seq2SeqConfig)
    model_provider_factory = FixedTranslationPreTrainedModelProviderFactory()
    model = create_model_with_mock_pretrained_model(config, model_provider_factory)

    nmt_test.test(
        config=config,
        last=True,
        by_book=True,
        save_confidences=True,
        scorers={"bleu", "chrf3", "ter"},
        model=model,
    )

    check_last_checkpoint_was_used(model_provider_factory, exp_dir)
    check_predictions(exp_dir, model_provider_factory.translation)
    check_confidences(exp_dir)
    check_scores(exp_dir)
    check_verse_scores(exp_dir, model_provider_factory.translation)
    check_linear_regression(exp_dir)

    delete_generated_paths(exp_dir, TEST_OUTPUT_PATTERNS)


def check_last_checkpoint_was_used(
    model_provider_factory: FixedTranslationPreTrainedModelProviderFactory, exp_dir: Path
):
    expected_checkpoint_dir = exp_dir / "run" / f"checkpoint-{CHECKPOINT_STEP}"
    assert model_provider_factory.inference_model_names == [str(expected_checkpoint_dir)]


def check_predictions(exp_dir: Path, expected_translation: str):
    tokenized_predictions = read_lines(exp_dir / f"test.trg-predictions.txt.{CHECKPOINT_STEP}")
    assert len(tokenized_predictions) == TEST_SIZE
    for prediction in tokenized_predictions:
        # The tokenized predictions keep the target language token and the end-of-sequence token
        assert prediction.startswith("spa_Latn ")
        assert prediction.endswith(" </s>")

    predictions = read_lines(exp_dir / f"test.trg-predictions.detok.txt.{CHECKPOINT_STEP}")
    assert predictions == [expected_translation] * TEST_SIZE


def check_confidences(exp_dir: Path):
    # The confidences file has two header rows, and then a row of tokens and a row of scores for
    # each translated sentence. The first score of a sentence is its sequence score.
    confidences_path = exp_dir / f"test.trg-predictions.txt.{CHECKPOINT_STEP}.confidences.tsv"
    rows = [line.split("\t") for line in read_lines(confidences_path)]
    assert len(rows) == 2 + 2 * TEST_SIZE
    assert rows[0][0] == "Sequence Number"
    assert rows[1][0] == "Sequence Score"

    for sentence_index in range(TEST_SIZE):
        token_row = rows[2 + 2 * sentence_index]
        score_row = rows[3 + 2 * sentence_index]
        assert token_row[0] == str(sentence_index + 1)
        assert float(score_row[0]) == expected_confidence(sentence_index)


def check_scores(exp_dir: Path):
    scores = read_scores(exp_dir / f"scores-{CHECKPOINT_STEP}.csv")

    # The first row scores all of the test sentences, and, because the test step was run with
    # the --by-book option, it is followed by a row for each book that they come from.
    overall_score = scores[0]
    assert overall_score["book"] == "ALL"
    assert overall_score["draft_index"] == "1"
    assert overall_score["src_iso"] == "en"
    assert overall_score["trg_iso"] == "es"
    assert int(overall_score["sent_len"]) == TEST_SIZE
    for scorer in ["BLEU", "chrF3"]:
        assert 0 <= float(overall_score[scorer]) <= 100
    assert float(overall_score["TER"]) >= 0

    # The overall confidence is the geometric mean of the sentence confidences
    expected_overall_confidence = math.exp(
        sum(mock_sequence_log_prob(sentence_index) for sentence_index in range(TEST_SIZE)) / TEST_SIZE
    )
    assert float(overall_score["Confidence"]) == pytest.approx(expected_overall_confidence, abs=1e-6)

    book_scores = scores[1:]
    assert len(book_scores) > 0
    assert all(book_score["book"] != "ALL" for book_score in book_scores)
    assert sum(int(book_score["sent_len"]) for book_score in book_scores) == TEST_SIZE


def check_verse_scores(exp_dir: Path, expected_translation: str):
    verse_scores = read_scores(exp_dir / f"test.trg-predictions.detok.txt.{CHECKPOINT_STEP}.scores.tsv", delimiter="\t")
    assert len(verse_scores) == TEST_SIZE

    for sentence_index, verse_score in enumerate(verse_scores):
        assert verse_score["Verse"] == str(sentence_index + 1)
        assert verse_score["Prediction"] == expected_translation
        assert verse_score["Reference"] != ""
        assert float(verse_score["Confidence"]) == expected_confidence(sentence_index)


def check_linear_regression(exp_dir: Path):
    # The confidence-to-chrF3 line of best fit is only calculated when both scorers are used
    with (exp_dir / f"linregress.{CHECKPOINT_STEP}.json").open("r", encoding="utf-8") as file:
        linear_regression = json.load(file)
    assert isinstance(linear_regression["slope"], float)
    assert isinstance(linear_regression["intercept"], float)


def expected_confidence(sentence_index: int):
    # The mock model's scores are float32 values, and the scores in the verse scores file are
    # rounded to eight decimal places, so the confidences are compared approximately.
    return pytest.approx(math.exp(mock_sequence_log_prob(sentence_index)), abs=1e-6)


def read_scores(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))
