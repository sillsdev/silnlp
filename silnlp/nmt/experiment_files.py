from pathlib import Path
from typing import Iterable, List, TextIO

from ..common.corpus import write_corpus
from .corpora import BASIC_DATA_PROJECT
from .corpus_inventory import CorpusInventory


class ExperimentFiles:
    _DATA_SET_PATTERNS = ("train.*.txt", "val.*.txt", "test.*.txt", "dict.*.txt")

    def __init__(self, exp_dir: Path, inventory: CorpusInventory, multi_ref_eval: bool) -> None:
        self._exp_dir = exp_dir
        self._inventory = inventory
        self._multi_ref_eval = multi_ref_eval

    def train_source(self) -> Path:
        return self._exp_dir / "train.src.txt"

    def train_source_detokenized(self) -> Path:
        return self._exp_dir / "train.src.detok.txt"

    def train_target(self) -> Path:
        return self._exp_dir / "train.trg.txt"

    def train_target_detokenized(self) -> Path:
        return self._exp_dir / "train.trg.detok.txt"

    def train_vref(self) -> Path:
        return self._exp_dir / "train.vref.txt"

    def validation_source(self) -> Path:
        return self._exp_dir / "val.src.txt"

    def validation_source_detokenized(self) -> Path:
        return self._exp_dir / "val.src.detok.txt"

    def validation_vref(self) -> Path:
        return self._exp_dir / "val.vref.txt"

    def validation_target(self, index: int = 0) -> Path:
        return self._exp_dir / (f"val.trg.txt.{index}" if self._multi_ref_eval else "val.trg.txt")

    def validation_target_detokenized(self, index: int = 0) -> Path:
        return self._exp_dir / (f"val.trg.detok.txt.{index}" if self._multi_ref_eval else "val.trg.detok.txt")

    def test_source(self, src_iso: str, trg_iso: str) -> Path:
        return self._exp_dir / f"{self._test_prefix(src_iso, trg_iso)}.src.txt"

    def test_source_detokenized(self, src_iso: str, trg_iso: str) -> Path:
        return self._exp_dir / f"{self._test_prefix(src_iso, trg_iso)}.src.detok.txt"

    def test_vref(self, src_iso: str, trg_iso: str) -> Path:
        return self._exp_dir / f"{self._test_prefix(src_iso, trg_iso)}.vref.txt"

    def test_target(self, src_iso: str, trg_iso: str, project: str = BASIC_DATA_PROJECT) -> Path:
        suffix = f".{project}" if self._inventory.has_multiple_test_projects(src_iso, trg_iso) else ""
        return self._exp_dir / f"{self._test_prefix(src_iso, trg_iso)}.trg.detok{suffix}.txt"

    def dictionary_source(self) -> Path:
        return self._exp_dir / "dict.src.txt"

    def dictionary_target(self) -> Path:
        return self._exp_dir / "dict.trg.txt"

    def dictionary_vref(self) -> Path:
        return self._exp_dir / "dict.vref.txt"

    def validation_reference_count(self, src_iso: str, trg_iso: str) -> int:
        if self._multi_ref_eval:
            return self._inventory.validation_project_count(src_iso, trg_iso)
        return 1

    def statistics_report(self) -> Path:
        return self._exp_dir / "tokenization_stats.csv"

    def statistics_spreadsheet(self) -> Path:
        return self.statistics_report().with_suffix(".xlsx")

    def tokenized_source_files(self) -> List[Path]:
        return sorted(self._exp_dir.glob("*.src.txt"))

    def tokenized_target_files(self) -> List[Path]:
        return sorted(self._exp_dir.glob("*.trg.txt"))

    def detokenized_source_files(self) -> List[Path]:
        return sorted(self._exp_dir.glob("*.src.detok.txt"))

    def detokenized_target_files(self) -> List[Path]:
        return sorted(self._exp_dir.glob("*.trg.detok.txt"))

    def delete_data_sets(self) -> None:
        for pattern in self._DATA_SET_PATTERNS:
            for path in self._exp_dir.glob(pattern):
                path.unlink()

    def append(self, path: Path, sentences: Iterable[str]) -> None:
        write_corpus(path, sentences, append=True)

    def fill(self, path: Path, count: int) -> None:
        write_corpus(path, ("" for _ in range(count)), append=True)

    def open_for_append(self, path: Path) -> TextIO:
        return path.open("a", encoding="utf-8", newline="\n")

    def _test_prefix(self, src_iso: str, trg_iso: str) -> str:
        if self._inventory.spans_multiple_test_iso_pairs():
            return f"test.{src_iso}.{trg_iso}"
        return "test"
