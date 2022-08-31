from typing import Iterable, List, Sequence

from ..common.stemmer import Stemmer


class NullStemmer(Stemmer):
    def train(self, corpus: Iterable[Sequence[str]]) -> None:
        pass

    def stem(self, words: Sequence[str]) -> Sequence[str]:
        return words
