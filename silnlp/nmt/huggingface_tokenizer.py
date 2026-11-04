import re
from typing import Dict, List, Optional, Set, Union

from sacremoses import MosesPunctNormalizer
from tokenizers import NormalizedString, Regex
from transformers.models.nllb.tokenization_nllb import NllbTokenizer
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..common.utils import Side
from .pretrained_tokenizer import PretrainedTokenizer
from .tokenizer import Tokenizer


class PunctuationNormalizingTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self._wrapped_tokenizer = tokenizer
        self._tokenizer = tokenizer._tokenizer
        self._mpn = MosesPunctNormalizer()
        self._mpn.substitutions = [(re.compile(r), sub) for r, sub in self._mpn.substitutions]
        self._pad_token = tokenizer.pad_token

    def __getattr__(self, name: str):
        return getattr(self._wrapped_tokenizer, name)

    def _normalize_text(self, text: Union[str, List[str], List[List[str]]]) -> Union[str, List[str], List[List[str]]]:
        if isinstance(text, str):
            return self._mpn.normalize(text)
        if isinstance(text, (list, tuple)) and len(text) > 0:
            if isinstance(text[0], (list, tuple)) and len(text[0]) > 0:
                return [[self._mpn.normalize(item) for item in row] for row in text]
            return [self._mpn.normalize(item) for item in text]
        return text

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]] = None,
        text_pair: Union[str, List[str], List[List[str]]] = None,
        text_target: Union[str, List[str], List[List[str]]] = None,
        text_pair_target: Union[str, List[str], List[List[str]]] = None,
        **kwargs,
    ) -> BatchEncoding:
        if text is None:
            raise ValueError('"text" input to PunctuationNormalizingTokenizer cannot be None')

        return self._wrapped_tokenizer(self._normalize_text(text), **kwargs)


class HuggingFaceTokenizer(Tokenizer):
    def __init__(
        self,
        pretrained: PretrainedTokenizer,
        lang_codes: Dict[str, str],
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self._pretrained = pretrained
        self._mpn = MosesPunctNormalizer()
        self._mpn.substitutions = [(re.compile(r), sub) for r, sub in self._mpn.substitutions]
        self._lang_codes = lang_codes
        self._max_source_length = max_source_length
        self._max_target_length = max_target_length
        self._special_tokens_of: Optional[PreTrainedTokenizerBase] = None
        self._special_tokens: Set[str] = set()

    @property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        """Whichever tokenizer is current: extending the vocabulary replaces it with one read back
        from disk, so holding on to a particular instance would leave this one a version behind."""
        return self._pretrained.load()

    @property
    def _all_special_tokens(self) -> Set[str]:
        tokenizer = self._tokenizer
        if self._special_tokens_of is not tokenizer:
            self._special_tokens_of = tokenizer
            self._special_tokens = set(tokenizer.all_special_tokens)
        return self._special_tokens

    def set_src_lang(self, src_lang: str) -> None:
        self._tokenizer.src_lang = self._lang_codes.get(src_lang, src_lang)

    def set_trg_lang(self, trg_lang: str) -> None:
        self._tokenizer.tgt_lang = self._lang_codes.get(trg_lang, trg_lang)

    def tokenize(
        self,
        side: Side,
        line: str,
        add_dummy_prefix: bool = True,
        sample_subwords: bool = False,
        add_special_tokens: bool = True,
    ) -> str:
        if isinstance(self._tokenizer, NllbTokenizer):
            line = self._mpn.normalize(line)
        if not add_dummy_prefix:
            line = "\ufffc" + line
        if side == Side.SOURCE:
            max_length = self._max_source_length
            if isinstance(self._tokenizer, T5Tokenizer):
                line = self._tokenizer.tgt_lang + " " + line
                max_length += 1
            if not add_dummy_prefix:
                max_length += 2
            tokens = self._tokenizer(
                line, add_special_tokens=add_special_tokens, max_length=max_length, truncation=True
            ).tokens()
        else:
            max_length = self._max_target_length
            if not add_dummy_prefix:
                max_length += 2
            tokens = self._tokenizer(
                text_target=line,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=True,
            ).tokens()
        if not add_dummy_prefix:
            tokens.remove("▁")
            tokens.remove("\ufffc")
        return " ".join(t.strip() for t in tokens)

    def normalize_normalized_string(self, line: NormalizedString) -> None:
        if isinstance(self._tokenizer, NllbTokenizer):
            line.replace(Regex(".+"), self._mpn.normalize(str(line.normalized)))
        self._tokenizer.backend_tokenizer.normalizer.normalize(line)

    def normalize(self, side: Side, line: str) -> str:
        if isinstance(self._tokenizer, NllbTokenizer):
            line = self._mpn.normalize(line)
        return self._tokenizer.backend_tokenizer.normalizer.normalize_str(line)

    def detokenize(self, line: str) -> str:
        tokens = line.split()
        tokens = [p for p in tokens if p not in self._all_special_tokens]
        return self._tokenizer.clean_up_tokenization(self._tokenizer.convert_tokens_to_string(tokens))


class CustomNormalizerWrapper:
    def __init__(self, tokenizer: HuggingFaceTokenizer) -> None:
        self._tokenizer = tokenizer

    def normalize(self, line: NormalizedString) -> None:
        self._tokenizer.normalize_normalized_string(line)
