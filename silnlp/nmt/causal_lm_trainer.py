from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .batch_size import find_executable_batch_size

LABEL_PAD_TOKEN_ID = -100


@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = LABEL_PAD_TOKEN_ID
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        pad_token_id = self.tokenizer.pad_token_id
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []
        for feature in features:
            ids = feature["input_ids"]
            mask = feature.get("attention_mask", [1] * len(ids))
            label = feature["labels"]
            pad_len = max_length - len(ids)
            # Right padding for training.
            input_ids.append(ids + [pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(label + [self.label_pad_token_id] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SilCausalTrainer(Trainer):
    def __init__(self, *args, auto_grad_acc: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_grad_acc = auto_grad_acc

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        if self._auto_grad_acc and args is not None:
            args.auto_find_batch_size = True
            inner_training_loop = find_executable_batch_size(super()._inner_training_loop, batch_size, self.accelerator)
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        return super()._inner_training_loop(
            batch_size=batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )
