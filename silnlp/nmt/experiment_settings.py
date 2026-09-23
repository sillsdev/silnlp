from typing import List, Optional

from ..common.utils import NoiseMethod, create_noise_methods
from .model_name import ModelName


class EvaluationSettings:
    """The evaluation section of the config. An experiment with no validation split has nothing to
    evaluate against, so evaluation is turned off rather than left to fail during training."""

    def __init__(self, evaluation: dict) -> None:
        self._evaluation = evaluation

    def metric_for_best_model(self) -> Optional[str]:
        metric = self._evaluation["metric_for_best_model"]
        return metric.lower() if metric is not None else None

    def detokenizes(self) -> bool:
        return self._evaluation["detokenize"]

    def early_stopping(self) -> Optional[dict]:
        return self._evaluation["early_stopping"]

    def disable_unless(self, has_validation_split: bool) -> None:
        if has_validation_split:
            return
        self._evaluation["eval_strategy"] = "no"
        self._evaluation["load_best_model_at_end"] = False
        self._evaluation["early_stopping"] = None
        self._evaluation["metric_for_best_model"] = None


class ScoringSettings:
    """How the test step scores translations against its references."""

    _DEFAULT_SACREBLEU_TOKENIZER = "13a"

    def __init__(self, data: dict) -> None:
        self._data = data

    def sacrebleu_tokenizer(self) -> str:
        return self._data.get("sacrebleu_tokenize", self._DEFAULT_SACREBLEU_TOKENIZER)


class TrainingSettings:
    """The training section of the config, with the adjustments the chosen model and the automatic
    gradient accumulation setting force on whatever was configured."""

    _MADLAD_SEQUENCE_LENGTH = 256
    _AUTOMATIC_BATCH_SIZE = 64

    def __init__(self, training: dict) -> None:
        self._training = training

    def fit_to(self, model_name: ModelName) -> None:
        if model_name.is_madlad():
            self._training["max_source_length"] = self._MADLAD_SEQUENCE_LENGTH
            self._training["max_target_length"] = self._MADLAD_SEQUENCE_LENGTH
        if self._training["auto_grad_acc"]:
            self._training["per_device_train_batch_size"] = self._AUTOMATIC_BATCH_SIZE
            self._training["gradient_accumulation_steps"] = 1


class TrainerSettings:
    """The settings of the training run itself, beyond the arguments handed to the trainer."""

    def __init__(self, training: dict) -> None:
        self._training = training

    def source_noise(self) -> List[NoiseMethod]:
        return create_noise_methods(self._training.get("src_noise", []))

    def samples_sequentially(self) -> bool:
        return self._training.get("sequential_sampling", False)

    def accumulates_gradient_automatically(self) -> bool:
        return self._training.get("auto_grad_acc", False)

    def checkpoints_gradients(self) -> bool:
        return self._training["gradient_checkpointing"]
