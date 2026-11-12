import logging
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional, Set

from tqdm import tqdm

from ..common.corpus import split_corpus
from ..common.utils import Side, add_tags_to_sentence
from .alignment_scores import AlignmentScores
from .corpora import BASIC_DATA_PROJECT, BasicParallelCorpus, CorpusPair, DataFile
from .corpus_inventory import CorpusInventory
from .data_set_files import DataSetFile, DataSetFileOpener, SplitFiles
from .experiment_files import ExperimentFiles
from .sentence_noiser import SentenceNoiser
from .tokenizer import Tokenizer

LOGGER = logging.getLogger(__name__)


class AlignmentQuality:
    """Which sentence pairs of a corpus align too poorly to train or validate on."""

    def __init__(
        self,
        alignment_scores: AlignmentScores,
        threshold: float,
        corpus: BasicParallelCorpus,
        source_name: str,
        target_name: str,
    ) -> None:
        self._alignment_scores = alignment_scores
        self._threshold = threshold
        self._corpus = corpus
        self._source_name = source_name
        self._target_name = target_name

    def poorly_aligned(self, test_indices: Set[int]) -> Set[int]:
        scored = self._corpus.to_data_frame()
        self._alignment_scores.add_to(scored, self._source_name, self._target_name)
        # The test set is chosen before alignment is considered, so its pairs are never excluded.
        candidates = scored.drop(list(test_indices))
        well_aligned = self._alignment_scores.filter(candidates, self._threshold)
        return set(candidates.index.difference(well_aligned.index))


class CorpusSplit:
    """Which sentences of a corpus go to the test, validation and training sets."""

    def __init__(self, pair: CorpusPair, corpus_size: int, quality: Optional[AlignmentQuality]) -> None:
        self._pair = pair
        self._test_indices: Optional[Set[int]] = set()
        self._validation_indices: Optional[Set[int]] = set()
        self._train_indices: Optional[Set[int]] = set()
        self._poorly_aligned: Set[int] = set()

        if pair.is_test:
            test_size = pair.size if pair.test_size is None else pair.test_size
            self._test_indices = split_corpus(corpus_size, test_size)
        if quality is not None and self._test_indices is not None:
            self._poorly_aligned = quality.poorly_aligned(self._test_indices)
        if pair.is_val and self._test_indices is not None:
            validation_size = pair.size if pair.val_size is None else pair.val_size
            self._validation_indices = split_corpus(
                corpus_size, validation_size, self._test_indices | self._poorly_aligned
            )
        if pair.is_train and self._test_indices is not None and self._validation_indices is not None:
            untaken = set(range(corpus_size)) - (self._test_indices | self._validation_indices | self._poorly_aligned)
            self._train_indices = split_corpus(untaken, pair.size)

    def is_test(self, index: int) -> bool:
        return self._pair.is_test and self._selects(self._test_indices, index)

    def is_validation(self, index: int) -> bool:
        return self._pair.is_val and self._usable(index) and self._selects(self._validation_indices, index)

    def is_train(self, index: int) -> bool:
        return self._pair.is_train and self._usable(index) and self._selects(self._train_indices, index)

    def _usable(self, index: int) -> bool:
        return index not in self._poorly_aligned

    def _selects(self, indices: Optional[Set[int]], index: int) -> bool:
        # No indices at all means the split was not narrowed down and takes every sentence.
        return indices is None or index in indices


class BasicDataSetFiles:
    """The data set files one basic corpus pair is written to."""

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
        opener = DataSetFileOpener(self._stack)
        train_blank: List[DataSetFile] = []
        validation_blank: List[DataSetFile] = []
        test_blank: List[DataSetFile] = []
        if self._inventory.has_scripture_data():
            # Basic sentences have no verse reference, but the scripture pairs of the same experiment
            # write one per line, so these files have to stay the same length.
            train_blank = opener.open(self._files.train_vref())
            validation_blank = opener.open(self._files.validation_vref(), *self._other_validation_targets())
            test_blank = opener.open(self._files.test_vref(self._src_iso, self._trg_iso), *self._other_test_targets())

        self._train = SplitFiles(
            opener.open(
                self._files.train_source(),
                self._files.train_target(),
                self._files.train_source_detokenized(),
                self._files.train_target_detokenized(),
            ),
            train_blank,
        )
        self._validation = SplitFiles(
            opener.open(
                self._files.validation_source(),
                self._files.validation_source_detokenized(),
                self._files.validation_target(),
                self._files.validation_target_detokenized(),
            ),
            validation_blank,
        )
        self._test = SplitFiles(
            opener.open(
                self._files.test_source(self._src_iso, self._trg_iso),
                self._files.test_source_detokenized(self._src_iso, self._trg_iso),
                self._files.test_target(self._src_iso, self._trg_iso),
            ),
            test_blank,
        )
        self._dictionary = SplitFiles(
            opener.open(self._files.dictionary_source(), self._files.dictionary_target()),
            opener.open(self._files.dictionary_vref()),
        )
        return self

    def __exit__(self, *exception) -> None:
        self._stack.close()

    def write_test(self, tokenized_source: str, source: str, target: str) -> None:
        self._test.write_row(tokenized_source, source, target)

    def write_validation(self, tokenized_source: str, source: str, tokenized_target: str, target: str) -> None:
        self._validation.write_row(tokenized_source, source, tokenized_target, target)

    def write_train(self, source_variants: List[str], target_variants: List[str], source: str, target: str) -> None:
        for source_variant, target_variant in zip(source_variants, target_variants):
            self._train.write_row(source_variant, target_variant, source, target)

    def write_dictionary(self, source_variants: List[str], target_variants: List[str]) -> None:
        self._dictionary.write_row("\t".join(source_variants), "\t".join(target_variants))

    def _other_validation_targets(self) -> List[Path]:
        indices = range(1, self._files.validation_reference_count(self._src_iso, self._trg_iso))
        return [self._files.validation_target(index) for index in indices] + [
            self._files.validation_target_detokenized(index) for index in indices
        ]

    def _other_test_targets(self) -> List[Path]:
        if not self._inventory.has_multiple_test_projects(self._src_iso, self._trg_iso):
            return []
        return [
            self._files.test_target(self._src_iso, self._trg_iso, project)
            for project in self._inventory.test_projects(self._src_iso, self._trg_iso)
            if project != BASIC_DATA_PROJECT
        ]


class BasicDataSetWriter:
    """Writes the data sets of a corpus pair that is plain parallel text rather than scripture."""

    def __init__(
        self,
        files: ExperimentFiles,
        inventory: CorpusInventory,
        tokenizer: Tokenizer,
        alignment_scores: AlignmentScores,
        mirror: bool,
    ) -> None:
        self._files = files
        self._inventory = inventory
        self._tokenizer = tokenizer
        self._alignment_scores = alignment_scores
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
        corpus = BasicParallelCorpus(src_file.path, trg_file.path)
        split = CorpusSplit(pair, corpus.count(), self._quality(pair, corpus, src_file, trg_file))
        noiser = SentenceNoiser(pair.src_noise)

        train_count = 0
        validation_count = 0
        test_count = 0
        dictionary_count = 0
        with BasicDataSetFiles(self._files, self._inventory, pair, src_file.iso, trg_file.iso) as data_set_files:
            for index, (source_line, target_line) in enumerate(tqdm(corpus)):
                source_sentence = add_tags_to_sentence(pair.tags, source_line)
                if split.is_test(index):
                    if src_file.include_test:
                        data_set_files.write_test(
                            self._tokenizer.tokenize(Side.SOURCE, source_sentence),
                            source_sentence,
                            self._tokenizer.normalize(Side.TARGET, target_line),
                        )
                        test_count += 1
                elif split.is_validation(index):
                    data_set_files.write_validation(
                        self._tokenizer.tokenize(Side.SOURCE, source_sentence),
                        source_sentence,
                        self._tokenizer.tokenize(Side.TARGET, target_line),
                        target_line,
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

        LOGGER.info(
            f"train size: {train_count}, val size: {validation_count}, "
            f"test size: {test_count}, dict size: {dictionary_count}"
        )
        return train_count

    def _quality(
        self, pair: CorpusPair, corpus: BasicParallelCorpus, src_file: DataFile, trg_file: DataFile
    ) -> Optional[AlignmentQuality]:
        if not (pair.is_train and pair.score_threshold > 0):
            return None
        return AlignmentQuality(
            self._alignment_scores, pair.score_threshold, corpus, src_file.path.stem, trg_file.path.stem
        )

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
        data_set_files.write_train(source_variants, target_variants, source, target)
        return len(source_variants)

    def _untagged_variants(self, side: Side, sentence: str) -> List[str]:
        return [
            self._tokenizer.tokenize(side, sentence, add_dummy_prefix=True, add_special_tokens=False),
            self._tokenizer.tokenize(side, sentence, add_dummy_prefix=False, add_special_tokens=False),
        ]
