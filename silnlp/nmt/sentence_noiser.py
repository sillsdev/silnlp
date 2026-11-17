from typing import Iterable, List

from ..common.utils import NoiseMethod


class SentenceNoiser:
    def __init__(self, noise_methods: Iterable[NoiseMethod]) -> None:
        self._noise_methods: List[NoiseMethod] = list(noise_methods)

    def apply(self, sentence: str) -> str:
        # Returned as given so that a sentence nobody noises keeps its original spacing.
        if len(self._noise_methods) == 0:
            return sentence
        tokens = sentence.split()
        for noise_method in self._noise_methods:
            tokens = noise_method(tokens)
        return " ".join(tokens)
