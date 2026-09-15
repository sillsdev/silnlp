import logging
from contextlib import ExitStack
from typing import List, Optional, Set, TextIO

from tqdm import tqdm

from ..common.corpus import split_corpus
from ..common.utils import Side, add_tags_to_sentence
from .corpora import BASIC_DATA_PROJECT, CorpusPair, DataFile, get_parallel_corpus_size
from .corpus_inventory import CorpusInventory
from .experiment_files import ExperimentFiles
from .sentence_noiser import SentenceNoiser
from .tokenizer import Tokenizer

LOGGER = logging.getLogger(__name__)


class CorpusSplit:
    """Which sentences of a corpus go to the test, validation and training sets."""

    def __init__(self, pair: CorpusPair, corpus_size: int) -> None:
        self._pair = pair
        self._test_indices: Optional[Set[int]] = set()
        self._validation_indices: Optional[Set[int]] = set()
        self._train_indices: Optional[Set[int]] = set()

        if pair.is_test:
            test_size = pair.size if pair.test_size is None else pair.test_size
            self._test_indices = split_corpus(corpus_size, test_size)
        if pair.is_val and self._test_indices is not None:
            validation_size = pair.size if pair.val_size is None else pair.val_size
            self._validation_indices = split_corpus(corpus_size, validation_size, self._test_indices)
        if pair.is_train and self._test_indices is not None and self._validation_indices is not None:
            self._train_indices = split_corpus(
                corpus_size, pair.size, self._test_indices | self._validation_indices
            )

    def is_test(self, index: int) -> bool:
        return self._pair.is_test and self._selects(self._test_indices, index)

    def is_validation(self, index: int) -> bool:
        return self._pair.is_val and self._selects(self._validation_indices, index)

    def is_train(self, index: int) -> bool:
        return self._pair.is_train and self._selects(self._train_indices, index)

    def _selects(self, indices: Optional[Set[int]], index: int) -> bool:
        # No indices at all means the split was not narrowed down and takes every sentence.
        return indices is None or index in indices


class BasicDataSetFiles:
    """The data set files one basic corpus pair is written to, opened together."""

    def __init__(
        self, files: ExperimentFiles, inventory: CorpusInventory, pair: CorpusPair, src_iso: str, trg_iso: str
    ) -> None:
        self._files = files
        self._inventory = inventory
        self._pair = pair
        self._src_iso = src_iso
        self._trg_iso = trg_iso
        self._stack = ExitStack()

    def __enter__(self) -> "BasicDataSetFiles":
        self._train_source = self._open(self._files.train_source())
        self._train_target = self._open(self._files.train_target())
        self._validation_source = self._open(self._files.validation_source())
        self._validation_target = self._open(self._files.validation_target())
        self._test_source = self._open(self._files.test_source(self._src_iso, self._trg_iso))
        self._test_target = self._open(self._files.test_target(self._src_iso, self._trg_iso))

        self._train_vref: Optional[TextIO] = None
        self._validation_vref: Optional[TextIO] = None
        self._test_vref: Optional[TextIO] = None
        self._other_test_targets: List[TextIO] = []
        self._other_validation_targets: List[TextIO] = []
        if self._inventory.has_scripture_data():
            self._open_scripture_aligned_files()

        self._dictionary_source: Optional[TextIO] = None
        self._dictionary_target: Optional[TextIO] = None
        self._dictionary_vref: Optional[TextIO] = None
        if self._pair.is_dictionary:
            self._dictionary_source = self._open(self._files.dictionary_source())
            self._dictionary_target = self._open(self._files.dictionary_target())
            self._dictionary_vref = self._open(self._files.dictionary_vref())
        return self

    def __exit__(self, *exception) -> None:
        self._stack.close()

    def write_test(self, source: str, target: str) -> None:
        self._test_source.write(source + "\n")
        self._test_target.write(target + "\n")
        self._write_blank(self._test_vref)
        for other in self._other_test_targets:
            other.write("\n")

    def write_validation(self, source: str, target: str) -> None:
        self._validation_source.write(source + "\n")
        self._validation_target.write(target + "\n")
        self._write_blank(self._validation_vref)
        for other in self._other_validation_targets:
            other.write("\n")

    def write_train(self, source_variants: List[str], target_variants: List[str]) -> None:
        for source, target in zip(source_variants, target_variants):
            self._train_source.write(source + "\n")
            self._train_target.write(target + "\n")
            self._write_blank(self._train_vref)

    def write_dictionary(self, source_variants: List[str], target_variants: List[str]) -> None:
        if self._dictionary_source is None or self._dictionary_target is None or self._dictionary_vref is None:
            return
        self._dictionary_source.write("\t".join(source_variants) + "\n")
        self._dictionary_target.write("\t".join(target_variants) + "\n")
        self._dictionary_vref.write("\n")

    def _open_scripture_aligned_files(self) -> None:
        # Basic sentences have no verse reference, but the scripture pairs of the same experiment
        # write one per line, so these files have to stay the same length.
        self._train_vref = self._open(self._files.train_vref())
        self._validation_vref = self._open(self._files.validation_vref())
        self._test_vref = self._open(self._files.test_vref(self._src_iso, self._trg_iso))
        if self._inventory.has_multiple_test_projects(self._src_iso, self._trg_iso):
            self._other_test_targets = [
                self._open(self._files.test_target(self._src_iso, self._trg_iso, project))
                for project in self._inventory.test_projects(self._src_iso, self._trg_iso)
                if project != BASIC_DATA_PROJECT
            ]
        self._other_validation_targets = [
            self._open(self._files.validation_target(index))
            for index in range(1, self._files.validation_reference_count(self._src_iso, self._trg_iso))
        ]

    def _write_blank(self, file: Optional[TextIO]) -> None:
        if file is not None:
            file.write("\n")

    def _open(self, path) -> TextIO:
        return self._stack.enter_context(self._files.open_for_append(path))


class BasicDataSetWriter:
    """Writes the data sets of a corpus pair that is plain parallel text rather than scripture."""

    def __init__(
        self, files: ExperimentFiles, inventory: CorpusInventory, tokenizer: Tokenizer, mirror: bool
    ) -> None:
        self._files = files
        self._inventory = inventory
        self._tokenizer = tokenizer
        self._mirror = mirror

    def write(self, pair: CorpusPair) -> int:
        train_count = 0
        for src_file, trg_file in zip(pair.src_files, pair.trg_files):
            train_count += self._write_file_pair(pair, src_file, trg_file)
        return train_count

    def _write_file_pair(self, pair: CorpusPair, src_file: DataFile, trg_file: DataFile) -> int:
        LOGGER.info(f"Preprocessing {src_file.path.stem} -> {trg_file.path.stem}")
        self._tokenizer.set_src_lang(src_file.iso)
        self._tokenizer.set_trg_lang(trg_file.iso)
        split = CorpusSplit(pair, get_parallel_corpus_size(src_file.path, trg_file.path))
        noiser = SentenceNoiser(pair.src_noise)

        train_count = 0
        validation_count = 0
        test_count = 0
        dictionary_count = 0
        with ExitStack() as stack:
            source_lines = stack.enter_context(src_file.path.open("r", encoding="utf-8"))
            target_lines = stack.enter_context(trg_file.path.open("r", encoding="utf-8"))
            data_set_files = stack.enter_context(
                BasicDataSetFiles(self._files, self._inventory, pair, src_file.iso, trg_file.iso)
            )

            index = 0
            for source_line, target_line in tqdm(zip(source_lines, target_lines)):
                source_line = source_line.strip()
                target_line = target_line.strip()
                if len(source_line) == 0 or len(target_line) == 0:
                    continue

                source_sentence = add_tags_to_sentence(pair.tags, source_line)
                if split.is_test(index):
                    data_set_files.write_test(
                        self._tokenizer.tokenize(Side.SOURCE, source_sentence),
                        self._tokenizer.normalize(Side.TARGET, target_line),
                    )
                    test_count += 1
                elif split.is_validation(index):
                    data_set_files.write_validation(
                        self._tokenizer.tokenize(Side.SOURCE, source_sentence),
                        self._tokenizer.tokenize(Side.TARGET, target_line),
                    )
                    validation_count += 1
                elif split.is_train(index):
                    train_count += self._write_train(
                        data_set_files, pair, noiser, src_file, trg_file, source_line, target_line
                    )

                if pair.is_dictionary:
                    data_set_files.write_dictionary(
                        self._untagged_variants(Side.SOURCE, source_sentence),
                        self._untagged_variants(Side.TARGET, target_line),
                    )
                    dictionary_count += 1

                index += 1

        LOGGER.info(
            f"train size: {train_count}, val size: {validation_count}, "
            f"test size: {test_count}, dict size: {dictionary_count}"
        )
        return train_count

    def _write_train(
        self,
        data_set_files: BasicDataSetFiles,
        pair: CorpusPair,
        noiser: SentenceNoiser,
        src_file: DataFile,
        trg_file: DataFile,
        source_line: str,
        target_line: str,
    ) -> int:
        written = self._write_train_sentence_pair(
            data_set_files, add_tags_to_sentence(pair.tags, noiser.apply(source_line)), target_line, pair
        )
        if not self._mirror:
            return written

        self._tokenizer.set_src_lang(trg_file.iso)
        self._tokenizer.set_trg_lang(src_file.iso)
        written += self._write_train_sentence_pair(
            data_set_files, add_tags_to_sentence(pair.tags, noiser.apply(target_line)), source_line, pair
        )
        self._tokenizer.set_src_lang(src_file.iso)
        self._tokenizer.set_trg_lang(trg_file.iso)
        return written

    def _write_train_sentence_pair(
        self, data_set_files: BasicDataSetFiles, source: str, target: str, pair: CorpusPair
    ) -> int:
        source_variants = [self._tokenizer.tokenize(Side.SOURCE, source, add_dummy_prefix=True)]
        target_variants = [self._tokenizer.tokenize(Side.TARGET, target, add_dummy_prefix=True)]
        if pair.is_lexical_data:
            source_variants.append(self._tokenizer.tokenize(Side.SOURCE, source, add_dummy_prefix=False))
            target_variants.append(self._tokenizer.tokenize(Side.TARGET, target, add_dummy_prefix=False))
        data_set_files.write_train(source_variants, target_variants)
        return len(source_variants)

    def _untagged_variants(self, side: Side, sentence: str) -> List[str]:
        return [
            self._tokenizer.tokenize(side, sentence, add_dummy_prefix=True, add_special_tokens=False),
            self._tokenizer.tokenize(side, sentence, add_dummy_prefix=False, add_special_tokens=False),
        ]
