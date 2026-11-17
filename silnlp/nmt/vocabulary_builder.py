from abc import ABC, abstractmethod

from .tokenizer import NullTokenizer, Tokenizer


class VocabularyBuilder(ABC):
    @abstractmethod
    def build(self, stats: bool = False) -> Tokenizer:
        """Settle the experiment's vocabulary and hand back the tokenizer to use from then on."""


class NoVocabularyBuilder(VocabularyBuilder):
    """Models that come with their own tokenizer, or corpora written as raw text, extend no vocabulary."""

    def build(self, stats: bool = False) -> Tokenizer:
        return NullTokenizer()
