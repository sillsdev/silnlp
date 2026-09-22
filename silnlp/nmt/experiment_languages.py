from typing import Dict, Set

from .corpus_inventory import CorpusInventory
from .prompt_messages import Language


class ExperimentLanguages:
    """The languages an experiment translates between, under the names the model knows them by. An
    experiment names its corpora by ISO code; a model may want something else, which lang_codes maps."""

    def __init__(self, lang_codes: Dict[str, str], inventory: CorpusInventory) -> None:
        self._lang_codes = lang_codes
        self._inventory = inventory

    def name_of(self, iso: str) -> str:
        return self._lang_codes.get(iso, iso)

    def of(self, iso: str) -> Language:
        return Language(iso=iso, name=self.name_of(iso))

    def validation_source(self) -> str:
        return self.name_of(self._inventory.default_validation_source_iso())

    def validation_target(self) -> str:
        return self.name_of(self._inventory.default_validation_target_iso())

    def test_source(self) -> str:
        return self.name_of(self._inventory.default_test_source_iso())

    def test_target(self) -> str:
        return self.name_of(self._inventory.default_test_target_iso())

    def training_source_iso(self) -> str:
        return self._inventory.default_test_source_iso() or self._any(self._inventory.source_isos())

    def training_target_iso(self) -> str:
        return self._inventory.default_test_target_iso() or self._any(self._inventory.target_isos())

    def _any(self, isos: Set[str]) -> str:
        return next(iter(isos)) if len(isos) > 0 else ""
