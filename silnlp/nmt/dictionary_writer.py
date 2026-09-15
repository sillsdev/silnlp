from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import Dict, List, Tuple

from ..common.corpus import Term, get_terms
from ..common.environment import SilNlpEnv
from ..common.utils import Side
from .corpora import DataFile
from .experiment_files import ExperimentFiles
from .terms import GlossLanguage, TermCategories
from .tokenizer import Tokenizer


class DictionaryWriter(ABC):
    @abstractmethod
    def write(
        self,
        src_terms_files: List[Tuple[DataFile, List[str]]],
        trg_terms_files: List[Tuple[DataFile, List[str]]],
    ) -> int:
        ...


class NoDictionaryWriter(DictionaryWriter):
    """Models that are given no dictionary at inference time write none."""

    def write(
        self,
        src_terms_files: List[Tuple[DataFile, List[str]]],
        trg_terms_files: List[Tuple[DataFile, List[str]]],
    ) -> int:
        return 0


class TermDictionaryWriter(DictionaryWriter):
    """Every rendering of every term, tokenized with and without the dummy prefix, one term per line."""

    def __init__(
        self,
        files: ExperimentFiles,
        tokenizer: Tokenizer,
        categories: TermCategories,
        gloss_language: GlossLanguage,
        environment: SilNlpEnv,
    ) -> None:
        self._files = files
        self._tokenizer = tokenizer
        self._categories = categories
        self._gloss_language = gloss_language
        self._environment = environment

    def write(
        self,
        src_terms_files: List[Tuple[DataFile, List[str]]],
        trg_terms_files: List[Tuple[DataFile, List[str]]],
    ) -> int:
        dict_count = 0
        with ExitStack() as stack:
            # The files are created even when nothing is written, because later steps read them unconditionally.
            target_file = stack.enter_context(self._files.open_for_append(self._files.dictionary_target()))
            vref_file = stack.enter_context(self._files.open_for_append(self._files.dictionary_vref()))

            if self._categories.excludes_everything():
                return 0
            categories = self._categories.as_set()
            gloss_iso = self._gloss_language.iso()

            for trg_terms_file, trg_terms in self._read(trg_terms_files, gloss_iso):
                self._tokenizer.set_trg_lang(trg_terms_file.iso)
                for trg_term in trg_terms.values():
                    if categories is not None and trg_term.cat not in categories:
                        continue
                    dict_count += self._write_term(target_file, vref_file, trg_term, trg_term.renderings)

            if self._gloss_language.is_available():
                all_src_terms = self._read(src_terms_files, gloss_iso)
                self._tokenizer.set_trg_lang(gloss_iso)
                for _, src_terms in all_src_terms:
                    for src_term in src_terms.values():
                        if categories is not None and src_term.cat not in categories:
                            continue
                        dict_count += self._write_term(target_file, vref_file, src_term, src_term.glosses)
        return dict_count

    def _write_term(self, target_file, vref_file, term: Term, phrases: List[str]) -> int:
        variants: List[str] = []
        for phrase in phrases:
            variants.append(
                self._tokenizer.tokenize(Side.TARGET, phrase, add_dummy_prefix=True, add_special_tokens=False)
            )
            variants.append(
                self._tokenizer.tokenize(Side.TARGET, phrase, add_dummy_prefix=False, add_special_tokens=False)
            )
        if len(variants) == 0:
            return 0
        target_file.write("\t".join(variants) + "\n")
        vref_file.write("\t".join(str(vref) for vref in term.vrefs) + "\n")
        return 1

    def _read(
        self, terms_files: List[Tuple[DataFile, List[str]]], gloss_iso
    ) -> List[Tuple[DataFile, Dict[str, Term]]]:
        return [
            (terms_file, get_terms(terms_file.path, iso=gloss_iso, environment=self._environment))
            for terms_file, _ in terms_files
        ]
