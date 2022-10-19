import abc
from typing import Any, Dict, Iterable, Sequence


class Stemmer(abc.ABC):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def train(self, corpus: Iterable[Sequence[str]]) -> None:
        pass

    @abc.abstractmethod
    def stem(self, words: Sequence[str]) -> Sequence[str]:
        pass
