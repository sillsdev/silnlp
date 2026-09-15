from typing import Dict, List, Optional, Set, Tuple

from ..common.corpus import Term, get_terms, get_terms_corpus, get_terms_data_frame
from ..common.environment import SilNlpEnv
from .corpora import DataFile
from .terms import GlossLanguage, TermCategories
from .terms_data_set import TermsDataSet


class TermsWriter:
    """Pairs the term lists of an experiment with one another, and with their glosses."""

    def __init__(
        self,
        data_set: TermsDataSet,
        categories: TermCategories,
        gloss_language: GlossLanguage,
        filter_books: Optional[Set[int]],
        environment: SilNlpEnv,
    ) -> None:
        self._data_set = data_set
        self._categories = categories
        self._gloss_language = gloss_language
        self._filter_books = filter_books
        self._environment = environment

    def write(
        self,
        src_terms_files: List[Tuple[DataFile, List[str]]],
        trg_terms_files: List[Tuple[DataFile, List[str]]],
    ) -> int:
        self._collect(src_terms_files, trg_terms_files)
        return self._data_set.write()

    def _collect(
        self,
        src_terms_files: List[Tuple[DataFile, List[str]]],
        trg_terms_files: List[Tuple[DataFile, List[str]]],
    ) -> None:
        if self._categories.excludes_everything():
            return
        categories = self._categories.as_set()
        gloss_iso = self._gloss_language.iso()

        all_src_terms = self._read(src_terms_files, gloss_iso)
        all_trg_terms = self._read(trg_terms_files, gloss_iso)

        for src_terms_file, src_terms, tags in all_src_terms:
            for trg_terms_file, trg_terms, _ in all_trg_terms:
                if src_terms_file.iso == trg_terms_file.iso:
                    continue
                terms = get_terms_corpus(src_terms, trg_terms, categories, self._filter_books)
                terms["source_lang"] = src_terms_file.iso
                terms["target_lang"] = trg_terms_file.iso
                self._data_set.add(terms, tags)

        if not self._gloss_language.is_available():
            return
        if self._gloss_language.can_serve_as_target():
            for src_terms_file, src_terms, tags in all_src_terms:
                terms = get_terms_data_frame(src_terms, categories, self._filter_books)
                terms = terms.rename(columns={"rendering": "source", "gloss": "target"})
                terms["source_lang"] = src_terms_file.iso
                terms["target_lang"] = gloss_iso
                self._data_set.add(terms, tags)
        if self._gloss_language.can_serve_as_source():
            for trg_terms_file, trg_terms, tags in all_trg_terms:
                terms = get_terms_data_frame(trg_terms, categories, self._filter_books)
                terms = terms.rename(columns={"rendering": "target", "gloss": "source"})
                terms["source_lang"] = gloss_iso
                terms["target_lang"] = trg_terms_file.iso
                self._data_set.add(terms, tags)

    def _read(
        self, terms_files: List[Tuple[DataFile, List[str]]], gloss_iso: Optional[str]
    ) -> List[Tuple[DataFile, Dict[str, Term], List[str]]]:
        return [
            (terms_file, get_terms(terms_file.path, iso=gloss_iso, environment=self._environment), tags)
            for terms_file, tags in terms_files
        ]
