import logging
from pathlib import Path
from typing import Callable, Optional

from machine.translation import Trainer, TrainStats
from machine.utils.progress_status import ProgressStatus
from transformers import EarlyStoppingCallback, HfArgumentParser, TrainingArguments

from .causal_lm_tokenizer import CausalLMTokenizer
from .causal_lm_trainer import DataCollatorForCausalLM, SilCausalTrainer
from .checkpoints import CheckpointDirectory
from .experiment_settings import EvaluationSettings, TrainerSettings
from .finetuning import Finetuning
from .training_arguments import TrainingArgumentsMapping
from .training_data_sets import CausalLMTrainingDataSets

LOGGER = logging.getLogger(__name__)


class CausalLMTrainingRun(Trainer):
    """One fine-tuning run of a decoder-only model."""

    def __init__(
        self,
        provider,
        tokenizer: CausalLMTokenizer,
        data_sets: CausalLMTrainingDataSets,
        finetuning: Finetuning,
        evaluation: EvaluationSettings,
        trainer_settings: TrainerSettings,
        training_arguments: TrainingArgumentsMapping,
        torch_dtype: str,
        mixed_precision: bool,
        clearml_queue: Optional[str] = None,
    ) -> None:
        self._provider = provider
        self._tokenizer = tokenizer
        self._data_sets = data_sets
        self._finetuning = finetuning
        self._evaluation = evaluation
        self._trainer_settings = trainer_settings
        self._training_arguments = training_arguments
        self._torch_dtype = torch_dtype
        self._mixed_precision = mixed_precision
        self._clearml_queue = clearml_queue
        self._trainer: Optional[SilCausalTrainer] = None
        self._stats = TrainStats()

    @property
    def stats(self) -> TrainStats:
        return self._stats

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        """Runs the fine-tuning. Neither progress nor cancellation is reported yet; both are accepted so
        that this fits the interface the rest of the toolchain trains through."""
        training_args = self.training_arguments()
        tokenizer = self._tokenizer.load()
        tokenizer.padding_side = "right"

        model = self._finetuning.applied_to(self._provider.create_model_for_training())
        train_dataset = self._data_sets.training()
        eval_dataset = self._data_sets.validation()

        self._trainer = SilCausalTrainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForCausalLM(
                tokenizer, pad_to_multiple_of=8 if (training_args.fp16 or training_args.bf16) else None
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            auto_grad_acc=self._trainer_settings.accumulates_gradient_automatically(),
        )
        self._add_early_stopping_to(self._trainer)

        last_checkpoint = CheckpointDirectory(Path(training_args.output_dir)).latest()
        train_result = self._trainer.train(
            resume_from_checkpoint=str(last_checkpoint.path) if last_checkpoint else None
        )

        self._stats.train_corpus_size = len(train_dataset) if train_dataset is not None else 0
        self._stats.metrics.update(train_result.metrics)
        self._stats.metrics["train_samples"] = self._stats.train_corpus_size

    def save(self) -> None:
        if self._trainer is None:
            raise RuntimeError("There is nothing to save until the training run has been made.")
        metrics = dict(self._stats.metrics)
        self._trainer.log_metrics("train", metrics)
        self._trainer.save_metrics("train", metrics)
        self._trainer.save_state()

    def write_effective_config(self, path: Path) -> None:
        self._training_arguments.write_effective_config(path, self.training_arguments())

    def training_arguments(self) -> TrainingArguments:
        args = self._training_arguments.collect(
            {
                "bf16": self._mixed_precision and self._torch_dtype == "bfloat16",
                "fp16": self._mixed_precision and self._torch_dtype == "float16",
            },
            self._clearml_queue,
        )
        return HfArgumentParser(TrainingArguments).parse_dict(args)[0]

    def _add_early_stopping_to(self, trainer: SilCausalTrainer) -> None:
        early_stopping: Optional[dict] = self._evaluation.early_stopping()
        if early_stopping:
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping["steps"],
                    early_stopping_threshold=early_stopping["min_improvement"],
                )
            )
