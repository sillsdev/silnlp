from typing import Iterable, List

from ..common.stemmer import Stemmer


class NullStemmer(Stemmer):
    def train(self, corpus: Iterable[List[str]]) -> None:
        pass

    def stem(self, words: List[str]) -> List[str]:
        return words
