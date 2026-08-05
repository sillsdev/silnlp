from math import log

from silnlp.common.sentence_context import CONTEXT_END_TOKEN, CONTEXT_START_TOKEN
from silnlp.common.translation_data_structures import SentenceTranslation


def build_translation() -> SentenceTranslation:
    # A seq2seq draft of a 3 sentence window: the language token, one sentence of preceding context,
    # the marked sentence, and one sentence of following context.
    tokens = ["spa_Latn", "uno", CONTEXT_START_TOKEN, "dos", "tres", CONTEXT_END_TOKEN, "cuatro"]
    token_scores = [log(0.1), log(0.2), log(0.3), log(0.4), log(0.5), log(0.6), log(0.7)]
    return SentenceTranslation(
        f"uno {CONTEXT_START_TOKEN} dos tres {CONTEXT_END_TOKEN} cuatro", tokens, token_scores, log(0.25)
    )


def test_only_the_marked_sentence_is_kept():
    assert build_translation().extract_central_segment().get_translation() == "dos tres"


def test_tokens_and_scores_are_sliced_to_the_same_span():
    extracted = build_translation().extract_central_segment()
    assert extracted.join_tokens_for_confidence_file() == "dos\ttres"
    expected_sequence_score = (log(0.4) + log(0.5)) / 2
    assert extracted.get_sequence_confidence_score() is not None
    assert abs(extracted.get_sequence_confidence_score() - pow(2.718281828459045, expected_sequence_score)) < 1e-9


def test_the_language_token_is_not_dropped_twice():
    # tokens[0] is only the forced decoder start token before slicing.
    assert build_translation().extract_central_segment().join_tokens_for_test_file() == "dos tres"


def test_an_unmarked_translation_is_left_alone():
    translation = SentenceTranslation("sin marcadores", ["sin", "marcadores"], [log(0.5), log(0.5)], log(0.5))
    extracted = translation.extract_central_segment()
    assert extracted.get_translation() == "sin marcadores"
    assert extracted.join_tokens_for_confidence_file() == "sin\tmarcadores"


def test_text_is_still_extracted_when_the_tokens_are_not_marked():
    # The decoder-only path reports the whole translation as a single token, so there is no token
    # span to slice, but the text can still be reduced to the marked sentence.
    text = f"uno {CONTEXT_START_TOKEN} dos {CONTEXT_END_TOKEN} tres"
    translation = SentenceTranslation(text, [text], [log(0.5)], log(0.5), starts_with_special_token=False)
    extracted = translation.extract_central_segment()
    assert extracted.get_translation() == "dos"
    assert extracted.join_tokens_for_confidence_file() == text
