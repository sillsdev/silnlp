import abc
from typing import Any, Dict, Iterable, List


class Stemmer(abc.ABC):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def train(self, corpus: Iterable[List[str]]) -> None:
        pass

    @abc.abstractmethod
    def stem(self, words: List[str]) -> List[str]:
        pass
