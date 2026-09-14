import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef

from ..common.corpus import load_corpus
from ..common.environment import SilNlpEnv
from .corpus_inventory import CorpusInventory

LOGGER = logging.getLogger(__name__)


class SharedTestSet:
    """The test verses another experiment already chose, named by "use_test_set_from"."""

    def __init__(
        self, exp_name: str, inventory: CorpusInventory, environment: SilNlpEnv, current_exp_dir: Path
    ) -> None:
        self._exp_dir = environment.get_mt_exp_dir(exp_name)
        self._inventory = inventory
        self._environment = environment
        self._current_exp_dir = current_exp_dir

    def indices_by_iso_pair(self) -> Dict[Tuple[str, str], Set[int]]:
        vref_paths = list(self._exp_dir.glob("test*.vref.txt"))
        if len(vref_paths) == 0:
            self._warn_about_no_test_files()
            return {}
        row_of = self._rows_of_the_full_versification()
        return {self._iso_pair_of(path): self._indices_in(path, row_of) for path in vref_paths}

    def _warn_about_no_test_files(self) -> None:
        # Pointing an experiment at itself also produces no test vrefs, but far more cryptically.
        if Path.samefile(self._exp_dir, self._current_exp_dir):
            LOGGER.warning('The experiment specified in "use_test_set_from" is the same as the current experiment.')
        else:
            LOGGER.warning(
                'The experiment specified in "use_test_set_from" does not contain any files matching '
                '"test*.vref.txt".'
            )

    def _iso_pair_of(self, vref_path: Path) -> Tuple[str, str]:
        stem = vref_path.stem
        if stem == "test.vref":
            return self._inventory.default_test_source_iso(), self._inventory.default_test_target_iso()
        _, src_iso, trg_iso, _ = stem.split(".", maxsplit=4)
        return src_iso, trg_iso

    def _indices_in(self, vref_path: Path, row_of: Dict[str, int]) -> Set[int]:
        indices: Set[int] = set()
        for vref_str in load_corpus(vref_path):
            if vref_str == "":
                continue
            vref = VerseRef.from_string(vref_str, ORIGINAL_VERSIFICATION)
            if vref.has_multiple:
                vref.simplify()
            indices.add(row_of[str(vref)])
        return indices

    def _rows_of_the_full_versification(self) -> Dict[str, int]:
        rows: Dict[str, int] = {}
        for row, vref_str in enumerate(load_corpus(self._environment.assets_dir / "vref.txt")):
            if vref_str != "":
                rows[vref_str] = row
        return rows
