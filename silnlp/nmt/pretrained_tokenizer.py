from pathlib import Path
from typing import Optional

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES, NllbTokenizer
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .model_name import ModelName
from .tokenizer_settings import TokenizerSource


class PretrainedTokenizer:
    """The tokenizer an experiment uses, loaded once and shared by everything that needs it."""

    def __init__(
        self,
        source: TokenizerSource,
        model_name: ModelName,
        exp_dir: Path,
    ) -> None:
        self._source = source
        self._model_name = model_name
        self._exp_dir = exp_dir
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None

    def build(self) -> PreTrainedTokenizerBase:
        """The tokenizer to extend, converting a SentencePiece model the experiment carries if there is one."""
        if self._tokenizer is None:
            if self._source.holds_unconverted_sentence_piece_model():
                # Only the families below know how to be built from a bare SentencePiece model; for any
                # other the conversion yields nothing and the next line is where that surfaces.
                tokenizer = self._convert_sentence_piece_model()
            else:
                tokenizer = self._from_pretrained(self._source.path_for_building())
            tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
            self._tokenizer = tokenizer
        return self._tokenizer

    def load(self) -> PreTrainedTokenizerBase:
        """The tokenizer of an experiment that has already been through preprocessing."""
        if self._tokenizer is None:
            self._tokenizer = self._from_pretrained(self._source.path_for_loading())
            self._tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return self._tokenizer

    def reload_from_experiment(self) -> PreTrainedTokenizerBase:
        self._tokenizer = self._from_pretrained(str(self._exp_dir))
        return self._tokenizer

    def _convert_sentence_piece_model(self) -> Optional[PreTrainedTokenizerBase]:
        if self._model_name.is_nllb():
            # NllbTokenizer normally falls back to FAIRSEQ_LANGUAGE_CODES, but only when
            # additional_special_tokens is None. When loading from a SentencePiece model,
            # SentencePieceExtractor.extract always sets it to the control symbols in the model (<s> and
            # </s>), so the fallback never runs and the language codes have to be passed in explicitly.
            tokenizer = NllbTokenizer.from_pretrained(
                str(self._exp_dir), token=False, extra_special_tokens=FAIRSEQ_LANGUAGE_CODES
            )
            tokenizer.save_pretrained(str(self._exp_dir))
            return tokenizer
        if self._model_name.is_madlad():
            tokenizer = T5Tokenizer.from_pretrained(str(self._exp_dir), token=False)
            tokenizer.add_special_tokens({"extra_special_tokens": ["<s>"]}, replace_extra_special_tokens=False)
            tokenizer.save_pretrained(str(self._exp_dir))
            return tokenizer
        return None

    def _from_pretrained(self, model_name_or_path: str) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, token=False)
