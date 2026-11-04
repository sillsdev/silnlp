from typing import Dict

from .corpus_inventory import CorpusInventory


class ExperimentLanguages:
    """The languages an experiment translates between, under the names the model knows them by. An
    experiment names its corpora by ISO code; a model may want something else, which lang_codes maps."""

    def __init__(self, lang_codes: Dict[str, str], inventory: CorpusInventory) -> None:
        self._lang_codes = lang_codes
        self._inventory = inventory

    def name_of(self, iso: str) -> str:
        return self._lang_codes.get(iso, iso)

    def validation_source(self) -> str:
        return self.name_of(self._inventory.default_validation_source_iso())

    def validation_target(self) -> str:
        return self.name_of(self._inventory.default_validation_target_iso())

    def test_source(self) -> str:
        return self.name_of(self._inventory.default_test_source_iso())

    def test_target(self) -> str:
        return self.name_of(self._inventory.default_test_target_iso())
