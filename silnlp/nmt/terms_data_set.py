from contextlib import ExitStack
from typing import List, Optional

import pandas as pd

from ..common.utils import Side, add_tags_to_dataframe
from .experiment_files import ExperimentFiles
from .tokenizer import Tokenizer


class TermsDataSet:
    """The term renderings of an experiment, written as extra training sentences with no verse references."""

    def __init__(self, files: ExperimentFiles, tokenizer: Tokenizer, mirror: bool) -> None:
        self._files = files
        self._tokenizer = tokenizer
        self._mirror = mirror
        self._terms: Optional[pd.DataFrame] = None

    def add(self, terms: pd.DataFrame, tags: Optional[List[str]] = None) -> None:
        if self._mirror:
            reversed_terms = terms.rename(
                columns={
                    "source": "target",
                    "target": "source",
                    "source_lang": "target_lang",
                    "target_lang": "source_lang",
                }
            )
            self._append(add_tags_to_dataframe(tags, reversed_terms))
        self._append(add_tags_to_dataframe(tags, terms))

    def write(self) -> int:
        if self._terms is None:
            return 0
        terms = self._terms.drop_duplicates(subset=["source", "target"])

        train_count = 0
        with ExitStack() as stack:
            source_file = stack.enter_context(self._files.open_for_append(self._files.train_source()))
            target_file = stack.enter_context(self._files.open_for_append(self._files.train_target()))
            vref_file = stack.enter_context(self._files.open_for_append(self._files.train_vref()))
            source_detok_file = stack.enter_context(
                self._files.open_for_append(self._files.train_source_detokenized())
            )
            target_detok_file = stack.enter_context(
                self._files.open_for_append(self._files.train_target_detokenized())
            )

            for _, term in terms.iterrows():
                source_term = term["source"]
                target_term = term["target"]
                self._tokenizer.set_src_lang(term["source_lang"])
                self._tokenizer.set_trg_lang(term["target_lang"])

                source_variants = [
                    self._tokenizer.tokenize(Side.SOURCE, source_term, add_dummy_prefix=True),
                    self._tokenizer.tokenize(Side.SOURCE, source_term, add_dummy_prefix=False),
                ]
                target_variants = [
                    self._tokenizer.tokenize(Side.TARGET, target_term, add_dummy_prefix=True),
                    self._tokenizer.tokenize(Side.TARGET, target_term, add_dummy_prefix=False),
                ]
                for source_variant, target_variant in zip(source_variants, target_variants):
                    source_file.write(source_variant + "\n")
                    target_file.write(target_variant + "\n")
                    vref_file.write("\n")
                    train_count += 1

                source_detok_file.write(source_term + "\n")
                target_detok_file.write(target_term + "\n")
        return train_count

    def _append(self, terms: pd.DataFrame) -> None:
        self._terms = terms if self._terms is None else pd.concat([self._terms, terms], ignore_index=True)
