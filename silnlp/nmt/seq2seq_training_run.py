import logging
from pathlib import Path
from typing import Any, Callable, Optional

from datasets.utils import logging as datasets_logging
from machine.translation import Trainer, TrainStats
from machine.utils.progress_status import ProgressStatus
from transformers import EarlyStoppingCallback, HfArgumentParser, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as transformers_logging

from ..common.utils import ReplaceRandomToken
from .checkpoints import CheckpointDirectory
from .decoder_inputs import DecoderInputs
from .experiment_files import ExperimentFiles
from .experiment_languages import ExperimentLanguages
from .experiment_settings import EvaluationSettings, TrainerSettings
from .pretrained_model_loader import PretrainedModelLoader
from .pretrained_tokenizer import PretrainedTokenizer
from .seq2seq_trainer import DataCollatorForSeq2SeqNoising, SilSeq2SeqTrainer
from .training_arguments import TrainingArgumentsMapping
from .training_data_sets import Seq2SeqTrainingDataSets
from .translation_metrics import TranslationMetrics
from .translation_settings import CheckpointRetention

LOGGER = logging.getLogger(__name__)


class Seq2SeqTrainingRun(Trainer):
    """One training run of a sequence-to-sequence model: what it trains on, what it reports, and what it
    leaves behind in its checkpoints."""

    def __init__(
        self,
        model_loader: PretrainedModelLoader,
        pretrained_tokenizer: PretrainedTokenizer,
        files: ExperimentFiles,
        languages: ExperimentLanguages,
        evaluation: EvaluationSettings,
        trainer_settings: TrainerSettings,
        retention: CheckpointRetention,
        training_arguments: TrainingArgumentsMapping,
        is_t5: bool,
        mixed_precision: bool,
        clearml_queue: Optional[str] = None,
    ) -> None:
        self._model_loader = model_loader
        self._pretrained_tokenizer = pretrained_tokenizer
        self._files = files
        self._languages = languages
        self._evaluation = evaluation
        self._trainer_settings = trainer_settings
        self._retention = retention
        self._training_arguments = training_arguments
        self._is_t5 = is_t5
        self._mixed_precision = mixed_precision
        self._clearml_queue = clearml_queue
        self._trainer: Optional[SilSeq2SeqTrainer] = None
        self._stats = TrainStats()

    @property
    def stats(self) -> TrainStats:
        return self._stats

    def train(
        self,
        progress: Optional[Callable[[ProgressStatus], None]] = None,
        check_canceled: Optional[Callable[[], None]] = None,
    ) -> None:
        """Runs the training. Neither progress nor cancellation is reported yet; both are accepted so
        that this fits the interface the rest of the toolchain trains through."""
        training_args = self.training_arguments()
        self._raise_logging_to(training_args)

        model, tokenizer = self._model_loader.for_training(
            training_args,
            self._languages.validation_source() or self._languages.test_source(),
            self._languages.validation_target() or self._languages.test_target(),
        )

        data_sets = Seq2SeqTrainingDataSets(self._files, self._pretrained_tokenizer)
        train_dataset = data_sets.training(training_args)
        eval_dataset = data_sets.validation(training_args)

        metrics = TranslationMetrics(self._evaluation, self._pretrained_tokenizer)
        self._trainer = SilSeq2SeqTrainer(
            model,
            training_args,
            self._collator(model, tokenizer, training_args),
            train_dataset,
            eval_dataset,
            processing_class=tokenizer,
            compute_metrics=None if metrics.is_produced_by_the_trainer() else metrics.compute,
            sequential_sampling=self._trainer_settings.samples_sequentially(),
            auto_grad_acc=self._trainer_settings.accumulates_gradient_automatically(),
        )
        self._add_early_stopping_to(self._trainer)

        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        train_result = self._trainer.train(resume_from_checkpoint=last_checkpoint)

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

        written = CheckpointDirectory(Path(self._trainer.args.output_dir))
        if not self._retention.keeps_optimizer_state():
            written.discard_optimizer_state()
        if not self._retention.keeps_tokenizers():
            written.discard_tokenizers()

    def write_effective_config(self, path: Path) -> None:
        self._training_arguments.write_effective_config(path, self.training_arguments())

    def training_arguments(self) -> Seq2SeqTrainingArguments:
        args = self._training_arguments.collect(
            # For context on floating point precision, see https://github.com/sillsdev/silnlp/issues/647
            {
                "fp16": self._mixed_precision and not self._is_t5,
                "bf16": self._mixed_precision and self._is_t5,
                "tf32": self._mixed_precision,
            },
            self._clearml_queue,
        )
        return HfArgumentParser(Seq2SeqTrainingArguments).parse_dict(args)[0]

    def _raise_logging_to(self, training_args: Seq2SeqTrainingArguments) -> None:
        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers_logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        datasets_logging.set_verbosity(log_level)
        transformers_logging.set_verbosity(log_level)
        transformers_logging.enable_default_handler()
        transformers_logging.enable_explicit_format()

    def _collator(self, model: Any, tokenizer: Any, training_args: Any) -> DataCollatorForSeq2SeqNoising:
        src_noise = self._trainer_settings.source_noise()
        for noise_method in src_noise:
            if isinstance(noise_method, ReplaceRandomToken):
                noise_method.filler_token = tokenizer.convert_tokens_to_ids(noise_method.filler_token)
        return DataCollatorForSeq2SeqNoising(
            tokenizer,
            DecoderInputs(model),
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
            src_noise=src_noise,
        )

    def _add_early_stopping_to(self, trainer: SilSeq2SeqTrainer) -> None:
        early_stopping: Optional[dict] = self._evaluation.early_stopping()
        if early_stopping:
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping["steps"],
                    early_stopping_threshold=early_stopping["min_improvement"],
                )
            )
