from typing import Optional

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class CausalLMTokenizer:
    """The tokenizer of a decoder-only model, loaded once. Models that ship no padding token pad with
    their end-of-sequence token instead, which the collator masks out of the loss anyway."""

    def __init__(self, model: str, trust_remote_code: bool) -> None:
        self._model = model
        self._trust_remote_code = trust_remote_code
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None

    def load(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self._model, trust_remote_code=self._trust_remote_code)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
        return self._tokenizer
