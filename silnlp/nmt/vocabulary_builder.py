from abc import ABC, abstractmethod


class VocabularyBuilder(ABC):
    @abstractmethod
    def build(self, stats: bool = False) -> None:
        ...


class NoVocabularyBuilder(VocabularyBuilder):
    """Decoder-only models come with their own tokenizer, so there is no vocabulary to extend."""

    def build(self, stats: bool = False) -> None:
        return
