from typing import List, Optional, cast

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

    def _apply_noise(self, tokens):
        words = tokens_to_words(tokens, subword_token=self.subword_token, is_spacer=self.is_spacer)
        words = cast(tf.Tensor, words.to_tensor())
        tag: Optional[str] = None
        if self.has_lang_tag:
            tag = words[:1]
            words = words[1:]
        for noise in self.noises:
            words = noise(words)
        if tag is not None:
            words = tf.concat([tag, words], axis=0)
        tokens = tf.RaggedTensor.from_tensor(words, padding="").flat_values
        return tokens
