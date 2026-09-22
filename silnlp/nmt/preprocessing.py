import logging

from .config import Config

LOGGER = logging.getLogger(__name__)


class Preprocessing:
    """Everything that happens to an experiment's corpora before it can be trained: the vocabulary its
    tokenizer needs, and the data sets a training run reads."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def run(self, stats: bool = False, force_align: bool = False) -> None:
        self._verify_input_files_exist()
        if self._config.data["tokenize"]:
            self._config.create_vocabulary_builder().build(stats)
        # The vocabulary is extended first, because doing so replaces the tokenizer being wrapped here.
        tokenizer = self._config.create_tokenizer()
        self._config.create_corpus_writer(tokenizer, force_align).write(stats)
        LOGGER.info("Preprocessing completed")

    def _verify_input_files_exist(self) -> None:
        missing_files = self._config.inventory.missing_input_files()
        if len(missing_files) > 0:
            raise RuntimeError("These corpus files do not exist: " + ", ".join(str(f) for f in missing_files))
