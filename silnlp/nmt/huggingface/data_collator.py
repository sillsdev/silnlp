from typing import Any, List, Optional, Union

from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer
from transformers.utils import PaddingStrategy

from ...common.utils import NoiseMethod


class DataCollatorForSeq2SeqNoising:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[Any] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        src_noise: List[NoiseMethod] = [],
        return_tensors: str = "pt",
    ):
        self._data_collator = DataCollatorForSeq2Seq(
            tokenizer, model, padding, max_length, pad_to_multiple_of, label_pad_token_id, return_tensors
        )
        self._src_noise = src_noise

    def __call__(self, features, return_tensors=None):
        if len(self._src_noise) > 0:
            for feature in features:
                input_ids = feature["input_ids"][:-2]
                for noise_method in self._src_noise:
                    input_ids = noise_method(input_ids)
                feature["input_ids"] = input_ids + feature["input_ids"][-2:]
                feature["attention_mask"] = feature["attention_mask"][: len(feature["input_ids"])]

        return self._data_collator(features, return_tensors)
