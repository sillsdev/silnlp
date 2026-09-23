import logging

from .corpus_inventory import CorpusInventory
from .experiment_data_set_writer import ExperimentDataSetWriter
from .vocabulary_builder import VocabularyBuilder

LOGGER = logging.getLogger(__name__)


class Preprocessor:
    """Everything that happens to an experiment's corpora before it can be trained: the vocabulary its
    tokenizer needs, and the data sets a training run reads."""

    def __init__(
        self, inventory: CorpusInventory, vocabulary: VocabularyBuilder, data_set_writer: ExperimentDataSetWriter
    ) -> None:
        self._inventory = inventory
        self._vocabulary = vocabulary
        self._data_set_writer = data_set_writer

    def run(self, stats: bool = False) -> None:
        self._verify_input_files_exist()
        self._vocabulary.build(stats)
        self._data_set_writer.write(stats)
        LOGGER.info("Preprocessing completed")

    def _verify_input_files_exist(self) -> None:
        missing_files = self._inventory.missing_input_files()
        if len(missing_files) > 0:
            raise RuntimeError("These corpus files do not exist: " + ", ".join(str(f) for f in missing_files))
