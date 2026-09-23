from typing import List

import numpy as np
import pytest

from silnlp.nmt.experiment_settings import EvaluationSettings
from silnlp.nmt.translation_metrics import TranslationMetrics


class StandInTokenizer:
    """Decodes an id as the letter at that position of the alphabet, so predictions read plainly."""

    pad_token_id = 0
    all_special_ids = [0, 1]

    def batch_decode(self, rows, skip_special_tokens: bool) -> List[str]:
        return [" ".join(self._token(int(i)) for i in row if int(i) not in self.all_special_ids) for row in rows]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._token(token_id)

    def _token(self, token_id: int) -> str:
        return chr(ord("a") + token_id - 2) if token_id >= 2 else ""


class StandInPretrainedTokenizer:
    def load(self) -> StandInTokenizer:
        return StandInTokenizer()


def metrics_for(metric: str, detokenize: bool = True) -> TranslationMetrics:
    evaluation = EvaluationSettings(
        {"metric_for_best_model": metric, "detokenize": detokenize, "early_stopping": None}
    )
    return TranslationMetrics(evaluation, StandInPretrainedTokenizer())


def test_a_metric_the_trainer_already_reports_is_not_computed_here():
    assert metrics_for("loss").is_produced_by_the_trainer()
    assert metrics_for("eval_loss").is_produced_by_the_trainer()


def test_a_translation_metric_is_computed_here():
    assert not metrics_for("bleu").is_produced_by_the_trainer()


def test_no_configured_metric_is_neither_named_nor_left_to_the_trainer():
    metrics = metrics_for(None)

    assert metrics.name() == ""
    assert not metrics.is_produced_by_the_trainer()


def test_an_unsupported_metric_is_rejected_when_the_metrics_are_set_up():
    with pytest.raises(ValueError, match="nonsense is not a supported metric"):
        metrics_for("nonsense")


def test_the_metric_name_is_reported_as_configured():
    assert metrics_for("chrf3").name() == "chrf3"


def test_the_configured_metric_is_lower_cased():
    assert metrics_for("BLEU").name() == "bleu"


def test_the_score_is_reported_under_the_metrics_own_name():
    metrics = metrics_for("bleu")
    preds = np.array([[2, 3, 4]])
    labels = np.array([[2, 3, 4]])

    result = metrics.compute((preds, labels))

    assert "bleu" in result


def test_the_generated_length_is_reported_alongside_the_score():
    metrics = metrics_for("bleu")
    preds = np.array([[2, 3, 0]])
    labels = np.array([[2, 3, 0]])

    result = metrics.compute((preds, labels))

    # The padding token is not part of what the model generated.
    assert result["gen_len"] == 2


def test_a_perfect_prediction_scores_at_the_top():
    metrics = metrics_for("bleu")
    preds = np.array([[2, 3, 4, 5]])
    labels = np.array([[2, 3, 4, 5]])

    assert metrics.compute((preds, labels))["bleu"] == 100.0


def test_predictions_handed_over_as_a_tuple_are_unwrapped():
    metrics = metrics_for("bleu")
    preds = np.array([[2, 3, 4, 5]])
    labels = np.array([[2, 3, 4, 5]])

    assert metrics.compute(((preds, np.array([[0]])), labels))["bleu"] == 100.0


def test_label_padding_is_replaced_before_decoding():
    # -100 marks label padding, which cannot be decoded.
    metrics = metrics_for("bleu")
    preds = np.array([[2, 3, 4, 5]])
    labels = np.array([[2, 3, 4, -100]])

    assert metrics.compute((preds, labels))["bleu"] < 100.0


def test_scores_are_rounded_for_reporting():
    metrics = metrics_for("chrf3")
    preds = np.array([[2, 3, 4]])
    labels = np.array([[2, 3, 5]])

    result = metrics.compute((preds, labels))

    assert result["chrf3"] == round(result["chrf3"], 4)
