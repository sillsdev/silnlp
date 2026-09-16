from .model_name import ModelName


class EvaluationSettings:
    """The evaluation section of the config. An experiment with no validation split has nothing to
    evaluate against, so evaluation is turned off rather than left to fail during training."""

    def __init__(self, evaluation: dict) -> None:
        self._evaluation = evaluation

    def disable_unless(self, has_validation_split: bool) -> None:
        if has_validation_split:
            return
        self._evaluation["eval_strategy"] = "no"
        self._evaluation["load_best_model_at_end"] = False
        self._evaluation["early_stopping"] = None
        self._evaluation["metric_for_best_model"] = None


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
