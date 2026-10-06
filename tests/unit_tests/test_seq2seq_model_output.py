import math
from typing import Dict, List

import pytest

from silnlp.common.translation_data_structures import TranslatedDraft
from silnlp.nmt.seq2seq_config import ModelOutput


class VocabularyTokenizer:
    def __init__(self, vocabulary: Dict[int, str], pad_token_id: int):
        self._vocabulary = vocabulary
        self.pad_token_id = pad_token_id

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._vocabulary[token_id] for token_id in ids]


def create_nllb_like_tokenizer() -> VocabularyTokenizer:
    return VocabularyTokenizer({0: "<pad>", 2: "</s>", 7: "spa_Latn", 8: "▁hola", 9: "▁mundo"}, pad_token_id=0)


def test_sentence_translation_leaves_out_decoder_start_token_and_padding() -> None:
    model_output = ModelOutput("hola", [2, 7, 8, 2, 0, 0], [0.0, 0.0, -0.4, -0.2, 0.0, 0.0], -0.2)

    sentence_translation = model_output.convert_to_sentence_translation(create_nllb_like_tokenizer())

    assert sentence_translation.join_tokens_for_test_file() == "spa_Latn ▁hola </s>"
    assert sentence_translation.join_token_scores_for_confidence_file().split("\t")[1:] == [
        str(math.exp(score)) for score in [0.0, -0.4, -0.2]
    ]


def test_sentence_translation_leaves_out_decoder_start_token_that_is_the_pad_token() -> None:
    # T5-style models start decoding from the pad token.
    model_output = ModelOutput("hola", [0, 8, 2, 0], [0.0, -0.4, -0.2, 0.0], -0.3)

    sentence_translation = model_output.convert_to_sentence_translation(create_nllb_like_tokenizer())

    assert sentence_translation.join_tokens_for_test_file() == "▁hola </s>"


def test_sentence_translations_are_weighted_by_generated_token_count() -> None:
    tokenizer = create_nllb_like_tokenizer()
    short = ModelOutput("hola", [2, 7, 8, 2, 0, 0], [0.0] * 6, math.log(0.9))
    long = ModelOutput("hola mundo", [2, 7, 8, 9, 8, 2], [0.0] * 6, math.log(0.6))
    draft = TranslatedDraft(
        [short.convert_to_sentence_translation(tokenizer), long.convert_to_sentence_translation(tokenizer)]
    )

    assert draft.compute_overall_confidence() == pytest.approx((0.9**3 * 0.6**5) ** (1 / 8))
