import logging
from pathlib import Path
from typing import List, Tuple

from ..common.environment import SilNlpEnv
from .alignment_scores import AlignmentScores
from .basic_data_set_writer import BasicDataSetWriter
from .corpora import CorpusPair, DataFile
from .corpus_inventory import CorpusInventory
from .dictionary_writer import DictionaryWriter
from .experiment_files import ExperimentFiles
from .scripture_data_set_writer import ScriptureDataSetWriter
from .terms_writer import TermsWriter
from .tokenization_statistics import TokenizationStatistics
from .tokenizer import Tokenizer

LOGGER = logging.getLogger(__name__)

TermsFiles = List[Tuple[DataFile, List[str]]]


class TermsSettings:
    """What the terms section of the config asks for: extra training sentences, a dictionary, or neither."""

    def __init__(self, terms: dict) -> None:
        self._terms = terms

    def writes_training_sentences(self) -> bool:
        return bool(self._terms["train"])

    def writes_dictionary(self) -> bool:
        return bool(self._terms["dictionary"])

    def needs_term_files(self) -> bool:
        return self.writes_dictionary() or self.writes_training_sentences()


class ScriptureDataSetWriterFactory:
    """Builds the writer for each scripture corpus pair of an experiment."""

    def __init__(
        self,
        files: ExperimentFiles,
        inventory: CorpusInventory,
        tokenizer: Tokenizer,
        alignment_scores: AlignmentScores,
        environment: SilNlpEnv,
        exp_dir: Path,
        mirror: bool,
        multi_ref_eval: bool,
    ) -> None:
        self._files = files
        self._inventory = inventory
        self._tokenizer = tokenizer
        self._alignment_scores = alignment_scores
        self._environment = environment
        self._exp_dir = exp_dir
        self._mirror = mirror
        self._multi_ref_eval = multi_ref_eval

    def writer_for(self, pair: CorpusPair) -> ScriptureDataSetWriter:
        return ScriptureDataSetWriter(
            pair,
            self._files,
            self._inventory,
            self._tokenizer,
            self._alignment_scores,
            self._environment,
            self._exp_dir,
            mirror=self._mirror,
            multi_ref_eval=self._multi_ref_eval,
        )


class ExperimentDataSetWriter:
    """Writes an experiment's corpora into the data sets a training run reads."""

    def __init__(
        self,
        corpus_pairs: List[CorpusPair],
        files: ExperimentFiles,
        scripture_writer_factory: ScriptureDataSetWriterFactory,
        basic_writer: BasicDataSetWriter,
        terms_writer: TermsWriter,
        dictionary_writer: DictionaryWriter,
        terms: TermsSettings,
        tokenize: bool,
    ) -> None:
        self._corpus_pairs = corpus_pairs
        self._files = files
        self._scripture_writer_factory = scripture_writer_factory
        self._basic_writer = basic_writer
        self._terms_writer = terms_writer
        self._dictionary_writer = dictionary_writer
        self._terms = terms
        self._tokenize = tokenize

    def write(self, stats: bool) -> int:
        self._files.delete_data_sets()

        train_count = 0
        src_terms_files: TermsFiles = []
        trg_terms_files: TermsFiles = []
        for pair in self._corpus_pairs:
            if pair.is_scripture:
                train_count += self._scripture_writer_factory.writer_for(pair).write()
            else:
                train_count += self._basic_writer.write(pair)

            if self._terms.needs_term_files():
                src_terms_files.extend((file, pair.tags) for file in pair.src_terms_files)
                trg_terms_files.extend((file, pair.tags) for file in pair.trg_terms_files)

        if self._terms.writes_training_sentences():
            terms_count = self._terms_writer.write(src_terms_files, trg_terms_files)
            LOGGER.info(f"terms train size: {terms_count}")
            train_count += terms_count

        if self._terms.writes_dictionary():
            dict_count = self._dictionary_writer.write(src_terms_files, trg_terms_files)
            LOGGER.info(f"dictionary size: {dict_count}")

        if stats and self._tokenize:
            TokenizationStatistics(self._files).write()

        return train_count
