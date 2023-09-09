from abc import ABC, abstractmethod
from typing import Iterable

from ..common.utils import Side


class Tokenizer(ABC):
    @abstractmethod
    def set_src_lang(self, src_lang: str) -> None:
        ...

    @abstractmethod
    def set_trg_lang(self, trg_lang: str) -> None:
        ...

    @abstractmethod
    def tokenize(
        self,
        side: Side,
        line: str,
        add_dummy_prefix: bool = True,
        sample_subwords: bool = False,
        add_special_tokens: bool = True,
    ) -> str:
        ...

    @abstractmethod
    def normalize(self, side: Side, line: str) -> str:
        ...

    @abstractmethod
    def normalize_no_tokenization(self, line: str) -> str:
        ...    

    @abstractmethod
    def detokenize(self, line: str) -> str:
        ...

    def tokenize_all(
        self,
        side: Side,
        lines: Iterable[str],
        add_dummy_prefix: bool = True,
        sample_subwords: bool = False,
        add_special_tokens: bool = True,
    ) -> Iterable[str]:
        for line in lines:
            yield self.tokenize(side, line, add_dummy_prefix, sample_subwords, add_special_tokens)

    def normalize_all(self, side: Side, lines: Iterable[str]) -> Iterable[str]:
        for line in lines:
            yield self.normalize(side, line)
    
    def normalize_no_tokenization_all(self, lines: Iterable[str]) -> Iterable[str]:
        for line in lines:
            yield self.normalize_no_tokenization(line)

    def detokenize_all(self, lines: Iterable[str]) -> Iterable[str]:
        for line in lines:
            yield self.detokenize(line)


class NullTokenizer(Tokenizer):
    def set_src_lang(self, src_lang: str) -> None:
        ...

    def set_trg_lang(self, trg_lang: str) -> None:
        ...

    def tokenize(
        self,
        side: Side,
        line: str,
        add_dummy_prefix: bool = True,
        sample_subwords: bool = False,
        add_special_tokens: bool = True,
    ) -> str:
        return line

    def normalize(self, side: Side, line: str) -> str:
        return line

    def detokenize(self, line: str) -> str:
        return line
