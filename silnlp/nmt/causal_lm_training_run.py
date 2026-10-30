import logging
from pathlib import Path
from typing import Callable, Optional

from machine.translation import Trainer, TrainStats
from machine.utils.progress_status import ProgressStatus
from transformers import EarlyStoppingCallback, HfArgumentParser, TrainingArguments

from .causal_lm_provider import CausalLMModelSettings, CausalLMProvider
from .causal_lm_tokenizer import CausalLMTokenizer
from .causal_lm_trainer import DataCollatorForCausalLM, SilCausalTrainer
from .causal_lm_training_data import (
    CausalLMEncoder,
    CausalLMTrainingDataSets,
    InterleavedEpochCallback,
    InterleavedTrainDataset,
)
from .checkpoints import CheckpointDirectory
from .example_retrieval import ExamplePools
from .experiment_settings import EvaluationSettings, TrainerSettings
from .finetuning import Finetuning
from .training_arguments import TrainingArgumentsMapping

LOGGER = logging.getLogger(__name__)


class CausalLMTrainingArguments:
    # Mirrors TRAINING_ARGS_CONFIG_MAPPING in seq2seq_config.py but without the seq2seq-only
    # generation keys (generation_max_length, generation_num_beams, predict_with_generate).
    _CONFIG_KEYS = {
        "train": {
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "gradient_checkpointing_kwargs",
            "log_level",
            "logging_first_step",
            "logging_steps",
            "logging_strategy",
            "max_steps",
            "num_train_epochs",
            "output_dir",
            "per_device_train_batch_size",
            "save_steps",
            "save_strategy",
            "save_total_limit",
            "train_sampling_strategy",
        },
        "eval": {
            "eval_accumulation_steps",
            "eval_delay",
            "eval_steps",
            "eval_strategy",
            "greater_is_better",
            "load_best_model_at_end",
            "metric_for_best_model",
            "per_device_eval_batch_size",
        },
        "params": {
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "learning_rate",
            "lr_scheduler_type",
            "max_grad_norm",
            "optim",
            "warmup_steps",
            "weight_decay",
        },
    }

    def __init__(
        self,
        config_root: dict,
        model_settings: CausalLMModelSettings,
        mixed_precision: bool,
        clearml_queue: Optional[str] = None,
    ) -> None:
        self._mapping = TrainingArgumentsMapping(self._CONFIG_KEYS, config_root)
        self._model_settings = model_settings
        self._mixed_precision = mixed_precision
        self._clearml_queue = clearml_queue

    def create(self) -> TrainingArguments:
        args = self._mapping.collect(
            self._model_settings.precision_arguments(self._mixed_precision), self._clearml_queue
        )
        return HfArgumentParser(TrainingArguments).parse_dict(args)[0]

    def write_effective_config(self, path: Path) -> None:
        self._mapping.write_effective_config(path, self.create())


class CausalLMTrainingRun(Trainer):
    """One fine-tuning run of a decoder-only model."""

    def __init__(
        self,
        provider: CausalLMProvider,
        tokenizer: CausalLMTokenizer,
        data_sets: CausalLMTrainingDataSets,
        finetuning: Finetuning,
        evaluation: EvaluationSettings,
        trainer_settings: TrainerSettings,
        training_arguments: CausalLMTrainingArguments,
        example_pools: ExamplePools,
    ) -> None:
        self._provider = provider
        self._tokenizer = tokenizer
        self._data_sets = data_sets
        self._finetuning = finetuning
        self._evaluation = evaluation
        self._trainer_settings = trainer_settings
        self._training_arguments = training_arguments
        self._example_pools = example_pools
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
        self._example_pools.ensure_available()
        training_args = self._training_arguments.create()
        tokenizer = self._tokenizer.load()
        tokenizer.padding_side = "right"

        model = self._finetuning.applied_to(self._provider.create_model_for_training())

        collator = DataCollatorForCausalLM(
            tokenizer, pad_to_multiple_of=8 if (training_args.fp16 or training_args.bf16) else None
        )
        encoder = CausalLMEncoder(tokenizer, collator)
        train_dataset = self._data_sets.translations(encoder)
        eval_dataset = self._data_sets.validation(encoder)
        # Instruction data is mixed into training only
        train_dataset = self._data_sets.mixed_with_instructions(train_dataset, encoder, training_args)

        self._trainer = SilCausalTrainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            auto_grad_acc=self._trainer_settings.accumulates_gradient_automatically(),
        )
        if isinstance(train_dataset, InterleavedTrainDataset):
            self._trainer.add_callback(InterleavedEpochCallback(train_dataset))
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
        self._training_arguments.write_effective_config(path)

    def _add_early_stopping_to(self, trainer: SilCausalTrainer) -> None:
        early_stopping: Optional[dict] = self._evaluation.early_stopping()
        if early_stopping:
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping["steps"],
                    early_stopping_threshold=early_stopping["min_improvement"],
                )
            )
