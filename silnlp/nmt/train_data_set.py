import random
from contextlib import ExitStack
from typing import Dict, List, Optional

import pandas as pd

from ..common.utils import Side, add_tags_to_dataframe
from .experiment_files import ExperimentFiles
from .tokenizer import Tokenizer


class TrainDataSet:
    """The training corpus an experiment accumulates, one corpus pair at a time."""

    _SOURCE_COLUMN_PREFIX = "source_"

    def __init__(self, files: ExperimentFiles, tokenizer: Tokenizer, mixed_source: bool) -> None:
        self._files = files
        self._tokenizer = tokenizer
        self._mixed_source = mixed_source
        self._corpus: Optional[pd.DataFrame] = None
        self._project_languages: Dict[str, str] = {}

    def note_language(self, project: str, iso: str) -> None:
        self._project_languages[project] = iso

    def add(self, source_project: str, target_project: str, tags: List[str], corpus: pd.DataFrame) -> None:
        add_tags_to_dataframe(tags, corpus)
        if self._mixed_source:
            self._add_mixed_source(source_project, target_project, corpus)
        else:
            self._corpus = corpus if self._corpus is None else pd.concat([self._corpus, corpus], ignore_index=True)

    def write(self) -> int:
        if self._corpus is None or len(self._corpus) == 0:
            return 0
        self._corpus.fillna("", inplace=True)
        source_columns = [column for column in self._corpus.columns if column.startswith("source")]

        written = 0
        with ExitStack() as stack:
            source_file = stack.enter_context(self._files.open_for_append(self._files.train_source()))
            target_file = stack.enter_context(self._files.open_for_append(self._files.train_target()))
            vref_file = stack.enter_context(self._files.open_for_append(self._files.train_vref()))
            source_detokenized = stack.enter_context(
                self._files.open_for_append(self._files.train_source_detokenized())
            )
            target_detokenized = stack.enter_context(
                self._files.open_for_append(self._files.train_target_detokenized())
            )

            for _, row in self._corpus.iterrows():
                source_sentence = self._prepare_source(row, source_columns)
                self._tokenizer.set_trg_lang(row["target_lang"])
                target_sentence = row["target"]

                source_file.write(self._tokenizer.tokenize(Side.SOURCE, source_sentence) + "\n")
                target_file.write(self._tokenizer.tokenize(Side.TARGET, target_sentence) + "\n")
                vref_file.write(str(row["vref"]) + "\n")
                source_detokenized.write(source_sentence + "\n")
                target_detokenized.write(target_sentence + "\n")
                written += 1
        return written

    def _add_mixed_source(self, source_project: str, target_project: str, corpus: pd.DataFrame) -> None:
        corpus.drop("source_lang", axis=1, inplace=True, errors="ignore")
        corpus.rename(columns={"source": f"{self._SOURCE_COLUMN_PREFIX}{source_project}"}, inplace=True)
        corpus.set_index(
            pd.MultiIndex.from_tuples(
                ((target_project, index) for index in corpus.index), names=["trg_project", "index"]
            ),
            inplace=True,
        )
        self._corpus = corpus if self._corpus is None else self._corpus.combine_first(corpus)

    def _prepare_source(self, row: pd.Series, source_columns: List[str]) -> str:
        if not self._mixed_source:
            self._tokenizer.set_src_lang(row["source_lang"])
            return row["source"]
        # Every project that has this verse is a candidate, and one of them is picked at random.
        available = [column for column in source_columns if row[column] != ""]
        chosen = random.choice(available)
        self._tokenizer.set_src_lang(self._project_languages[chosen[len(self._SOURCE_COLUMN_PREFIX) :]])
        return row[chosen]
