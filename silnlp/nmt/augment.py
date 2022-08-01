from typing import List, Optional, Type, Tuple
from abc import ABC, abstractmethod
import logging
import sentencepiece as sp
from pathlib import Path

from .utils import encode_sp_lines, encode_sp
from ..common.corpus import write_corpus

LOGGER = logging.getLogger(__package__ + ".augment")


class AugmentMethod(ABC):
    @abstractmethod
    def __pre__(self, args):
        pass

    @abstractmethod
    def __augment_corpus__(self,
                           train_src_filename: Path,
                           train_trg_filename: Path,
                           train_vref_filename: Path,
                           src: List[str],
                           trg: List[str],
                           vref: List[str],
                           src_spp: Optional[sp.SentencePieceProcessor],
                           trg_spp: Optional[sp.SentencePieceProcessor]) -> int:
        pass

    @abstractmethod
    def __augment_sentence__(self, src: str, trg: str,
                             src_spp: Optional[sp.SentencePieceProcessor],
                             trg_spp: Optional[sp.SentencePieceProcessor]) -> Tuple[List[str], List[str]]:
        pass

    @abstractmethod
    def __post__(self, args):
        pass


# Creates supplemental copies of the src and trg text using a simple shift-cipher.  One supplemental copy is created
# for each 'key' value specified.  The 'key' value indicates the number of characters to shift.
"""
class CipherAugment(AugmentMethod):
    def __init__(self, args):
        LOGGER.warning("Not fully implemented - CipherAugment class")
        self.keys = [int(k) for k in args.get('keys', '').split(',')]

    def __pre__(self, args):
        pass

    def __augment_corpus__(self, args):
        pass

    def __augment_sentence__(self, args) -> int:
        pass

    def __post__(self, args):
        pass
"""


# Creates additional tokenized copies of the src and trg text using alternate subword encodings.
class SubwordAugment(AugmentMethod):
    def __init__(self, args):
        self.encodings = int(args.get('encodings', 0))

    def __pre__(self, args):
        pass

    def __augment_corpus__(self,
                           train_src_filename: Path, train_trg_filename: Path, train_vref_filename: Path,
                           src: List[str], trg: List[str], vref: List[str],
                           src_spp: Optional[sp.SentencePieceProcessor],
                           trg_spp: Optional[sp.SentencePieceProcessor]) -> int:
        augment_count = 0
        for encoding in range(self.encodings):
            write_corpus(train_src_filename, encode_sp_lines(src_spp, src, sample_subwords=True), append=True)
            write_corpus(train_trg_filename, encode_sp_lines(trg_spp, trg, sample_subwords=True), append=True)
            write_corpus(train_vref_filename, (str(vr) for vr in vref), append=True)
            augment_count += len(src)
        return augment_count

    def __augment_sentence__(self,
                             src: str, trg: str,
                             src_spp: Optional[sp.SentencePieceProcessor],
                             trg_spp: Optional[sp.SentencePieceProcessor]) -> Tuple[List[str], List[str]]:
        src_augments: List[str] = [encode_sp(src_spp, src, sample_subwords=True) for x in range(self.encodings)]
        trg_augments: List[str] = [encode_sp(trg_spp, trg, sample_subwords=True) for x in range(self.encodings)]
        return src_augments, trg_augments

    def __post__(self, args):
        pass


# Creates an additional copy of the src / trg text by transliterating to the specified script
"""
class TransliterateAugment(AugmentMethod):
    def __init__(self, args):
        LOGGER.warning("Not fully implemented - TransliterateAugment class")
        self.source = args.get('source', None)
        self.target = args.get('target', None)

    def __pre__(self, args):
        pass

    def __augment_corpus__(self, args):
        pass

    def __augment_sentence__(self, args) -> int:
        pass

    def __post__(self, args):
        pass
"""


# Creates an additional version of the src / trg text of the form (see https://aclanthology.org/2021.emnlp-main.263.pdf)
#     Source: <src> + <trg>
#     Target: <trg> + <src>
"""
class BiTAugment(AugmentMethod):
    def __init__(self, args):
        LOGGER.warning("Not fully implemented - BiTAugment class")

    def __pre__(self, args):
        pass

    def __augment_corpus__(self, args):
        pass

    def __augment_sentence__(self, args) -> int:
        pass

    def __post__(self, args):
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

    def __pre__(self, args):
        pass

    def __augment_corpus__(self, args):
        pass

    def __augment_sentence__(self, args) -> int:
        pass

    def __post__(self, args):
        pass
"""


def create_augment_methods(params: List[List]) -> List[AugmentMethod]:
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


