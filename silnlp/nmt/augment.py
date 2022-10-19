import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Type

import sentencepiece as sp

from ..common.corpus import write_corpus
from ..common.utils import Side
from .tokenizer import Tokenizer

LOGGER = logging.getLogger(__package__ + ".augment")


class AugmentMethod(ABC):
    @abstractmethod
    def pre(self, args):
        pass

    @abstractmethod
    def augment_sentence(self, src: str, trg: str, tokenizer: Tokenizer) -> Tuple[List[str], List[str]]:
        pass

    @abstractmethod
    def post(self, args):
        pass


# Creates supplemental copies of the src and trg text using a simple shift-cipher.  One supplemental copy is created
# for each 'key' value specified.  The 'key' value indicates the number of characters to shift.
"""
class CipherAugment(AugmentMethod):
    def __init__(self, args):
        LOGGER.warning("Not fully implemented - CipherAugment class")
        self.keys = [int(k) for k in args.get('keys', '').split(',')]

    def pre(self, args):
        pass

    def augment_sentence(self, args) -> int:
        pass

    def post(self, args):
        pass
"""


# Creates additional tokenized copies of the src and trg text using alternate subword encodings.
class SubwordAugment(AugmentMethod):
    def __init__(self, args):
        self.encodings = int(args.get("encodings", 0))

    def pre(self, args):
        pass

    def augment_sentence(self, src: str, trg: str, tokenizer: Tokenizer) -> Tuple[List[str], List[str]]:
        src_augments: List[str] = [
            tokenizer.tokenize(Side.SOURCE, src, sample_subwords=True) for _ in range(self.encodings)
        ]
        trg_augments: List[str] = [
            tokenizer.tokenize(Side.TARGET, trg, sample_subwords=True) for _ in range(self.encodings)
        ]
        return src_augments, trg_augments

    def post(self, args):
        pass


# Creates an additional copy of the src / trg text by transliterating to the specified script
"""
class TransliterateAugment(AugmentMethod):
    def __init__(self, args):
        LOGGER.warning("Not fully implemented - TransliterateAugment class")
        self.source = args.get('source', None)
        self.target = args.get('target', None)

    def pre(self, args):
        pass

    def augment_sentence(self, args) -> int:
        pass

    def post(self, args):
        pass
"""


# Creates an additional version of the src / trg text of the form (see https://aclanthology.org/2021.emnlp-main.263.pdf)
#     Source: <src> + <trg>
#     Target: <trg> + <src>
"""
class BiTAugment(AugmentMethod):
    def __init__(self, args):
        LOGGER.warning("Not fully implemented - BiTAugment class")

    def pre(self, args):
        pass

    def augment_sentence(self, args) -> int:
        pass

    def post(self, args):
        pass
"""


# Cross-lingual Language Modeling (see https://arxiv.org/abs/1901.07291)
"""
class TLMAugment(AugmentMethod):
    def __init__(self, args):
        LOGGER.warning("Not fully implemented - TLMAugment class")
        self.probability = float(args.get('probability', 0.15))
        self.alpha = float(args.get('alpha', 0.1))
        self.shuffle = args.get('shuffle', False)

    def pre(self, args):
        pass

    def augment_sentence(self, args) -> int:
        pass

    def post(self, args):
        pass
"""


def create_augment_methods(params: List[dict]) -> List[AugmentMethod]:
    methods: List[AugmentMethod] = []
    for module in params:
        augment_type, args = next(iter(module.items()))
        if not isinstance(args, list):
            args = [args]
        augment_type = augment_type.lower()
        augment_method_class: Type[AugmentMethod]
        if augment_type == "subword":
            augment_method_class = SubwordAugment
        else:
            raise ValueError("Invalid augment type: %s" % augment_type)
        methods.append(augment_method_class(*args))
    return methods
