from typing import List

from silnlp.common.utils import NoiseMethod
from silnlp.nmt.sentence_noiser import SentenceNoiser


class DropFirstToken(NoiseMethod):
    def __call__(self, tokens: List[str]) -> List[str]:
        return tokens[1:]


class Uppercase(NoiseMethod):
    def __call__(self, tokens: List[str]) -> List[str]:
        return [token.upper() for token in tokens]


def test_a_noiser_with_no_methods_returns_the_sentence_untouched():
    assert SentenceNoiser([]).apply("  two   spaces  ") == "  two   spaces  "


def test_a_single_noise_method_is_applied_to_the_tokens():
    assert SentenceNoiser([DropFirstToken()]).apply("one two three") == "two three"


def test_noise_methods_are_applied_in_order():
    assert SentenceNoiser([DropFirstToken(), Uppercase()]).apply("one two") == "TWO"


def test_noising_rejoins_the_tokens_with_single_spaces():
    assert SentenceNoiser([Uppercase()]).apply("one   two") == "ONE TWO"
