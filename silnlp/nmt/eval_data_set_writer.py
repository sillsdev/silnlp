import itertools
import random
from contextlib import ExitStack
from typing import List, Optional, TextIO, cast

import pandas as pd

from ..common.utils import Side
from .corpus_inventory import CorpusInventory
from .eval_data_set import EvalDataSet
from .experiment_files import ExperimentFiles
from .tokenizer import Tokenizer

_TARGET_COLUMN_PREFIX = "target_"


class ScriptureValidationSetWriter:
    """Writes the validation set, with one target file per reference when several are kept."""

    def __init__(self, files: ExperimentFiles, tokenizer: Tokenizer, multi_ref_eval: bool) -> None:
        self._files = files
        self._tokenizer = tokenizer
        self._multi_ref_eval = multi_ref_eval

    def write(self, validation: EvalDataSet) -> int:
        if validation.is_empty():
            return 0
        for (src_iso, trg_iso), corpus in validation.language_pairs():
            self._tokenizer.set_src_lang(src_iso)
            self._tokenizer.set_trg_lang(trg_iso)
            self._files.append(
                self._files.validation_source(), self._tokenizer.tokenize_all(Side.SOURCE, corpus["source"])
            )
            self._files.append(self._files.validation_source_detokenized(), corpus["source"])

        self._write_targets(validation, self._tokenizer)
        self._write_targets(validation, None)
        vrefs = itertools.chain.from_iterable(corpus["vref"] for _, corpus in validation.language_pairs())
        self._files.append(self._files.validation_vref(), (str(vref) for vref in vrefs))
        return validation.total_size()

    def _write_targets(self, validation: EvalDataSet, tokenizer: Optional[Tokenizer]) -> None:
        with ExitStack() as stack:
            reference_files: List[TextIO] = []
            for (src_iso, trg_iso), corpus in validation.language_pairs():
                if tokenizer is not None:
                    tokenizer.set_src_lang(src_iso)
                    tokenizer.set_trg_lang(trg_iso)
                columns = [column for column in corpus.columns if column.startswith("target")]
                if self._multi_ref_eval:
                    self._write_every_reference(stack, reference_files, corpus, columns, tokenizer, src_iso, trg_iso)
                else:
                    self._write_one_reference(stack, reference_files, corpus, columns, tokenizer)

    def _write_every_reference(
        self, stack, reference_files, corpus, columns, tokenizer, src_iso: str, trg_iso: str
    ) -> None:
        reference_count = self._files.validation_reference_count(src_iso, trg_iso)
        for index in corpus.index:
            for reference in range(reference_count):
                if len(reference_files) == reference:
                    # Both passes write here, so the plain text follows the tokenized text.
                    reference_files.append(
                        stack.enter_context(self._files.open_for_append(self._files.validation_target(reference)))
                    )
                if reference >= len(columns):
                    reference_files[reference].write("\n")
                    continue
                reference_files[reference].write(
                    self._rendered(corpus, index, columns[reference], tokenizer) + "\n"
                )

    def _write_one_reference(self, stack, reference_files, corpus, columns, tokenizer) -> None:
        for index in corpus.index:
            if len(reference_files) == 0:
                path = (
                    self._files.validation_target()
                    if tokenizer is not None
                    else self._files.validation_target_detokenized()
                )
                reference_files.append(stack.enter_context(self._files.open_for_append(path)))
            with_text = [column for column in columns if cast(str, corpus.loc[index, column]).strip() != ""]
            reference_files[0].write(self._rendered(corpus, index, random.choice(with_text), tokenizer) + "\n")

    def _rendered(self, corpus: pd.DataFrame, index, column: str, tokenizer: Optional[Tokenizer]) -> str:
        if tokenizer is None:
            return cast(str, corpus.loc[index, column])
        return tokenizer.tokenize(Side.TARGET, cast(str, corpus.loc[index, column]).strip())


class ScriptureTestSetWriter:
    """Writes the test set, with one target file per project that contributes a reference."""

    def __init__(self, files: ExperimentFiles, inventory: CorpusInventory, tokenizer: Tokenizer) -> None:
        self._files = files
        self._inventory = inventory
        self._tokenizer = tokenizer

    def write(self, test: EvalDataSet) -> int:
        written = 0
        for (src_iso, trg_iso), corpus in test.language_pairs():
            self._tokenizer.set_src_lang(src_iso)
            self._tokenizer.set_trg_lang(trg_iso)
            self._files.append(self._files.test_vref(src_iso, trg_iso), (str(vref) for vref in corpus["vref"]))
            self._files.append(
                self._files.test_source(src_iso, trg_iso),
                self._tokenizer.tokenize_all(Side.SOURCE, corpus["source"]),
            )
            self._files.append(self._files.test_source_detokenized(src_iso, trg_iso), corpus["source"])
            written += len(corpus)
            self._write_targets(src_iso, trg_iso, corpus)
        return written

    def _write_targets(self, src_iso: str, trg_iso: str, corpus: pd.DataFrame) -> None:
        remaining = self._inventory.test_projects(src_iso, trg_iso)
        for column in [column for column in corpus.columns if column.startswith("target")]:
            project = column[len(_TARGET_COLUMN_PREFIX) :]
            self._files.append(
                self._files.test_target(src_iso, trg_iso, project),
                self._tokenizer.normalize_all(Side.TARGET, corpus[column]),
            )
            remaining.remove(project)
        if self._inventory.has_multiple_test_projects(src_iso, trg_iso):
            for project in remaining:
                self._files.fill(self._files.test_target(src_iso, trg_iso, project), len(corpus))
