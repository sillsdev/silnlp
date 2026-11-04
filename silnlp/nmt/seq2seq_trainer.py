from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset
from torch import Tensor, nn, optim
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

from ..common.utils import NoiseMethod
from .batch_size import find_executable_batch_size
from .decoder_inputs import DecoderInputs


class DataCollatorForSeq2SeqNoising:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        decoder_inputs: DecoderInputs,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        src_noise: List[NoiseMethod] = [],
        return_tensors: str = "pt",
    ):
        # No model is passed on, so that the decoder inputs are built in one place rather than
        # depending on whether the model happens to carry a shift of its own.
        self._data_collator = DataCollatorForSeq2Seq(
            tokenizer, None, padding, max_length, pad_to_multiple_of, label_pad_token_id, return_tensors
        )
        self._decoder_inputs = decoder_inputs
        self._src_noise = src_noise

    def __call__(self, features, return_tensors=None):
        if len(self._src_noise) > 0:
            for feature in features:
                input_ids = feature["input_ids"][:-2]
                for noise_method in self._src_noise:
                    input_ids = noise_method(input_ids)
                feature["input_ids"] = input_ids + feature["input_ids"][-2:]
                feature["attention_mask"] = feature["attention_mask"][: len(feature["input_ids"])]

        batch = self._data_collator(features, return_tensors)
        if batch.get("labels") is not None:
            batch["decoder_input_ids"] = self._decoder_inputs.from_labels(batch["labels"])
        return batch


class SilSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[Seq2SeqTrainingArguments] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[optim.Optimizer], Optional[optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        sequential_sampling: bool = False,
        auto_grad_acc: bool = False,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self._sequential_sampling = sequential_sampling
        self._auto_grac_acc = auto_grad_acc

    def _get_train_sampler(self, train_dataset: Optional[TorchDataset] = None) -> Optional[Sampler]:
        if self._sequential_sampling:
            return None
        return super()._get_train_sampler(train_dataset)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        if self._auto_grac_acc:
            (args if args is not None else self.args).auto_find_batch_size = True
            inner_training_loop = find_executable_batch_size(super()._inner_training_loop, batch_size, self.accelerator)
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        else:
            return super()._inner_training_loop(
                batch_size=batch_size,
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
