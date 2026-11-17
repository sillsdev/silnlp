import logging
from typing import Any, Dict, Optional, Set

import evaluate
import numpy as np

from .experiment_settings import EvaluationSettings
from .pretrained_tokenizer import PretrainedTokenizer

LOGGER = logging.getLogger(__name__)


class TranslationMetrics:
    """The metric an evaluation pass reports. Some metrics the trainer already produces itself; the rest
    are computed here from the model's predictions."""

    # "loss" and "eval_loss" are both evaluation loss; the early stopping callback prefixes "eval_" to
    # any metric that does not already start with it.
    _PRODUCED_BY_THE_TRAINER = ["loss", "eval_loss"]
    _MODULES = {
        "bleu": "sacrebleu",
        "chrf3": "chrf",
        "chrf3+": "chrf",
        "chrf3++": "chrf",
        "m-bleu": "sacrebleu",
        "m-chrf3": "chrf",
        "m-chrf3+": "chrf",
        "m-chrf3++": "chrf",
    }

    def __init__(self, evaluation: EvaluationSettings, pretrained: PretrainedTokenizer) -> None:
        self._evaluation = evaluation
        self._pretrained = pretrained
        self._name = evaluation.metric_for_best_model() or ""
        self._module: Optional[str] = None
        self._metric: Any = None
        if self._name != "" and not self.is_produced_by_the_trainer():
            self._module = self._MODULES.get(self._name)
            if self._module is None:
                raise ValueError(f"{self._name} is not a supported metric.")
            self._metric = evaluate.load(self._module)

    def name(self) -> str:
        return self._name

    def is_produced_by_the_trainer(self) -> bool:
        return self._name in self._PRODUCED_BY_THE_TRAINER

    def compute(self, eval_preds) -> dict:
        tokenizer = self._pretrained.load()
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the labels as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        if self._evaluation.detokenizes():
            decoded_preds = [pred.strip() for pred in tokenizer.batch_decode(preds, skip_special_tokens=True)]
            decoded_labels = [[label.strip()] for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        else:
            special_ids = set(tokenizer.all_special_ids)
            decoded_preds = [self._joined(pred, tokenizer, special_ids) for pred in preds]
            decoded_labels = [[self._joined(label, tokenizer, special_ids)] for label in labels]

        result: Dict[str, Any] = {self._name: self._score(decoded_preds, decoded_labels)}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return {key: round(value, 4) for key, value in result.items()}

    def _joined(self, ids, tokenizer, special_ids: Set[int]) -> str:
        return " ".join(tokenizer.convert_ids_to_tokens(int(id)) for id in ids if id not in special_ids).strip()

    def _score(self, decoded_preds, decoded_labels) -> float:
        if self._name == "bleu":
            result = self._metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                lowercase=True,
                force=not self._evaluation.detokenizes(),
            )
        elif self._module == "chrf":
            result = self._metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                char_order=6,
                word_order=self._name.count("+"),
                beta=3,
                lowercase=True,
                eps_smoothing="+" in self._name,
            )
        return result["score"]
