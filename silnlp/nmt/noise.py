from typing import List, Optional, Tuple, Union, cast
from abc import ABC, abstractmethod
import random

import tensorflow as tf
from opennmt.data import Noise, WordNoiser, tokens_to_words


class SILWordNoiser(WordNoiser):
    def __init__(
        self,
        noises: Optional[List[Noise]] = None,
        subword_token: str = "￭",
        is_spacer: Optional[bool] = None,
        has_lang_tag: bool = False,
    ) -> None:
        super().__init__(noises=noises, subword_token=subword_token, is_spacer=is_spacer)
        self.has_lang_tag = has_lang_tag

    def _call(
        self, tokens: tf.Tensor, sequence_length: Optional[Union[int, tf.Tensor]], keep_shape: bool
    ) -> Tuple[tf.Tensor, Union[int, tf.Tensor]]:
        rank = tokens.shape.rank
        if rank == 1:
            input_length = tf.shape(tokens)[0]
            if sequence_length is not None:
                tokens = tokens[:sequence_length]
            else:
                tokens = tokens[: tf.math.count_nonzero(tokens)]
            words = tokens_to_words(tokens, subword_token=self.subword_token, is_spacer=self.is_spacer)
            words = cast(tf.Tensor, words.to_tensor())
            if self.has_lang_tag:
                tag = words[:1]
                words = words[1:]
            for noise in self.noises:
                words = noise(words)
            if self.has_lang_tag:
                words = tf.concat([tag, words], axis=0)
            outputs = tf.RaggedTensor.from_tensor(words, padding="").flat_values
            output_length = tf.shape(outputs)[0]
            if keep_shape:
                outputs = tf.pad(outputs, [[0, input_length - output_length]])
            return outputs, output_length
        else:
            return super()._call(tokens, sequence_length=sequence_length, keep_shape=keep_shape)


class NoiseMethod(ABC):
    @abstractmethod
    def __call__(self, tokens: List[str]) -> List[str]:
        pass


def random_bool(probability: float) -> bool:
    """Returns True with given probability

    Args:
        probability: probability to return True

    """
    assert 0 <= probability <= 1, "probability needs to be >= 0 and <= 1"
    return random.random() < probability


class DeleteRandomToken(NoiseMethod):
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if not random_bool(self.probability)]


class ReplaceRandomToken(NoiseMethod):
    def __init__(self, probability: float, filler_token: str = "<blank>") -> None:
        self.probability = probability
        self.filler_token = filler_token

    def __call__(self, tokens: List[str]) -> List[str]:
        new_tokens = tokens.copy()
        for i in range(len(new_tokens)):
            if random_bool(self.probability):
                new_tokens[i] = self.filler_token
        return new_tokens


class RandomTokenPermutation(NoiseMethod):
    def __init__(self, distance: int) -> None:
        self.distance = distance

    def __call__(self, tokens: List[str]) -> List[str]:
        new_indices = [i + random.uniform(0, self.distance + 1) for i in range(len(tokens))]
        return [x for _, x in sorted(zip(new_indices, tokens), key=lambda pair: pair[0])]
