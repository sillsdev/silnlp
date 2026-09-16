import logging
import random
from pathlib import Path
from typing import Iterable, Optional, Set, Union

import pandas as pd

from ..common.corpus import (
    exclude_chapters,
    get_scripture_parallel_corpus,
    include_chapters,
    split_corpus,
    split_parallel_corpus,
)
from ..common.environment import SilNlpEnv
from .alignment_scores import AlignmentScores
from .corpora import CorpusPair, DataFile, DataFileMapping, get_data_file_pairs
from .corpus_inventory import CorpusInventory
from .eval_data_set import EvalDataSet
from .eval_data_set_writer import ScriptureTestSetWriter, ScriptureValidationSetWriter
from .experiment_files import ExperimentFiles
from .sentence_noiser import SentenceNoiser
from .shared_test_set import SharedTestSet
from .tokenizer import Tokenizer
from .train_data_set import TrainDataSet

LOGGER = logging.getLogger(__name__)


class SplitIndices:
    """The verses set aside for testing and validation, settled by the first file pair that needs them and
    reused by the rest so that every file pair of a corpus pair is split the same way."""

    def __init__(
        self,
        disjoint_test: bool,
        disjoint_val: bool,
        test_size: Union[float, int],
        val_size: Union[float, int],
    ) -> None:
        self._disjoint_test = disjoint_test
        self._disjoint_val = disjoint_val
        self._test_size = test_size
        self._val_size = val_size
        self._test: Optional[Iterable[int]] = None
        self._val: Optional[Iterable[int]] = None

    def reserve_for_test(self, indices: Iterable[int]) -> None:
        if self._test is None:
            self._test = indices

    def for_test(self, available: Iterable[int], corpus_count: int) -> Optional[Iterable[int]]:
        if self._disjoint_test and self._test is None:
            remaining = set(available)
            if self._disjoint_val and self._val is not None:
                remaining.difference_update(self._val)
            self._test = self._sample(remaining, self._test_size, corpus_count)
        return self._test

    def for_validation(self, available: Iterable[int], corpus_count: int) -> Optional[Iterable[int]]:
        if self._disjoint_val and self._val is None:
            remaining = set(available)
            if self._disjoint_test and self._test is not None:
                remaining.difference_update(self._test)
            self._val = self._sample(remaining, self._val_size, corpus_count)
        return self._val

    def _sample(self, remaining: Set[int], size: Union[float, int], corpus_count: int) -> Set[int]:
        if isinstance(size, float):
            size = int(size if size > 1 else corpus_count * size)
        return set(random.sample(remaining, min(size, len(remaining))))


class ScriptureDataSetWriter:
    """Divides one scripture corpus pair into the training, validation and test data sets."""

    def __init__(
        self,
        pair: CorpusPair,
        files: ExperimentFiles,
        inventory: CorpusInventory,
        tokenizer: Tokenizer,
        alignment_scores: AlignmentScores,
        environment: SilNlpEnv,
        exp_dir: Path,
        mirror: bool,
        multi_ref_eval: bool,
    ) -> None:
        self._pair = pair
        self._files = files
        self._inventory = inventory
        self._tokenizer = tokenizer
        self._alignment_scores = alignment_scores
        self._environment = environment
        self._multi_ref_eval = multi_ref_eval
        self._mirror = mirror

        self._test_size = pair.size if pair.test_size is None else pair.test_size
        self._val_size = pair.size if pair.val_size is None else pair.val_size
        self._indices = SplitIndices(pair.disjoint_test, pair.disjoint_val, self._test_size, self._val_size)

        self._train = TrainDataSet(files, tokenizer, mixed_source=pair.mapping == DataFileMapping.MIXED_SRC)
        self._validation = EvalDataSet()
        self._test = EvalDataSet()
        if pair.use_test_set_from != "":
            self._test = EvalDataSet(
                SharedTestSet(pair.use_test_set_from, inventory, environment, exp_dir).indices_by_iso_pair()
            )

    def write(self) -> int:
        LOGGER.info(f"Preprocessing {self._describe(self._pair.src_files)} -> {self._describe(self._pair.trg_files)}")
        for src_file, trg_file in get_data_file_pairs(self._pair):
            self._add(src_file, trg_file)

        train_count = self._train.write()
        val_count = ScriptureValidationSetWriter(
            self._files, self._tokenizer, multi_ref_eval=self._multi_ref_eval
        ).write(self._validation)
        test_count = ScriptureTestSetWriter(self._files, self._inventory, self._tokenizer).write(self._test)
        LOGGER.info(f"train size: {train_count}, val size: {val_count}, test size: {test_count},")
        return train_count

    def _add(self, src_file: DataFile, trg_file: DataFile) -> None:
        pair = self._pair
        self._train.note_language(src_file.project, src_file.iso)
        self._train.note_language(trg_file.project, trg_file.iso)

        corpus = get_scripture_parallel_corpus(src_file.path, trg_file.path, environment=self._environment)
        if len(pair.src_noise) > 0:
            noiser = SentenceNoiser(pair.src_noise)
            corpus["source"] = [noiser.apply(sentence) for sentence in corpus["source"]]

        cur_train = self._training_chapters(corpus)
        corpus_count = len(cur_train)

        scored = pair.is_train and pair.score_threshold > 0
        if scored:
            self._alignment_scores.add_to(
                cur_train, f"{src_file.iso}-{src_file.project}", f"{trg_file.iso}-{trg_file.project}"
            )

        if pair.is_test:
            cur_train = self._split_off_test(corpus, cur_train, corpus_count, src_file, trg_file)

        if scored:
            cur_train = self._alignment_scores.filter(cur_train, pair.score_threshold)

        if pair.is_val:
            cur_train = self._split_off_validation(cur_train, corpus_count, src_file, trg_file)

        if pair.is_train:
            self._add_to_training(cur_train, src_file, trg_file)

    def _training_chapters(self, corpus: pd.DataFrame) -> pd.DataFrame:
        pair = self._pair
        if len(pair.corpus_books) > 0:
            selected = include_chapters(corpus, pair.corpus_books)
            return exclude_chapters(selected, pair.test_books) if len(pair.test_books) > 0 else selected
        if len(pair.test_books) > 0:
            return exclude_chapters(corpus, pair.test_books)
        return corpus

    def _split_off_test(
        self,
        corpus: pd.DataFrame,
        cur_train: pd.DataFrame,
        corpus_count: int,
        src_file: DataFile,
        trg_file: DataFile,
    ) -> pd.DataFrame:
        pair = self._pair
        cur_test = None
        if len(pair.test_books) > 0:
            cur_test = include_chapters(corpus, pair.test_books)
            self._indices.reserve_for_test(cur_test.index)

        test_indices = self._indices.for_test(cur_train.index, corpus_count)
        chosen = self._test.indices_for(src_file.iso, trg_file.iso, default=test_indices)
        if cur_test is not None:
            if self._test_size > 0:
                _, cur_test = split_parallel_corpus(cur_test, self._test_size, chosen)
        else:
            cur_train, cur_test = split_parallel_corpus(cur_train, self._test_size, chosen)

        cur_test.drop("score", axis=1, inplace=True, errors="ignore")
        if src_file.include_test:
            self._test.add(src_file.iso, trg_file.iso, trg_file.project, pair.tags, cur_test)
        return cur_train

    def _split_off_validation(
        self, cur_train: pd.DataFrame, corpus_count: int, src_file: DataFile, trg_file: DataFile
    ) -> pd.DataFrame:
        val_indices = self._indices.for_validation(cur_train.index, corpus_count)
        cur_train, cur_val = split_parallel_corpus(
            cur_train, self._val_size, self._validation.indices_for(src_file.iso, trg_file.iso, default=val_indices)
        )
        self._validation.add(src_file.iso, trg_file.iso, trg_file.project, self._pair.tags, cur_val)
        return cur_train

    def _add_to_training(self, cur_train: pd.DataFrame, src_file: DataFile, trg_file: DataFile) -> None:
        pair = self._pair
        cur_train["source_lang"] = src_file.iso
        cur_train["target_lang"] = trg_file.iso

        train_indices = split_corpus(set(cur_train.index), pair.size)
        _, cur_train = split_parallel_corpus(cur_train, pair.size, train_indices)

        if self._mirror:
            self._train.add(
                trg_file.project,
                src_file.project,
                pair.tags,
                cur_train.rename(
                    columns={
                        "source": "target",
                        "target": "source",
                        "source_lang": "target_lang",
                        "target_lang": "source_lang",
                    }
                ),
            )
        self._train.add(src_file.project, trg_file.project, pair.tags, cur_train)

    def _describe(self, data_files: Iterable[DataFile]) -> str:
        names = [data_file.path.stem for data_file in data_files]
        description = ", ".join(names)
        return f"[{description}]" if len(names) > 1 else description
