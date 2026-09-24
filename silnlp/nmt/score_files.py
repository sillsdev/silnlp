from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from machine.scripture import book_number_to_id

from ..common.translator import CONFIDENCE_SUFFIX
from .corpus_inventory import CorpusInventory

LINREGRESS_PREFIX = "linregress"
SINGLE_TEST_SET_PREFIX = "test"
VERSE_SCORES_SUFFIX = ".scores.tsv"


@dataclass(frozen=True)
class TestSetNames:
    source: str
    vref: str
    predictions: str
    predictions_detokenized: str
    references_pattern: str
    linregress: str
    src_iso: str
    trg_iso: str
    draft_index: int


@dataclass(frozen=True)
class ReferenceFiles:
    paths: List[Path]
    select_random_line: bool


class TestSetFiles:
    """One test set at one checkpoint step and draft index: what the checkpoint was given, what it
    produced, and what those predictions are scored against."""

    def __init__(self, exp_dir: Path, names: TestSetNames, inventory: CorpusInventory) -> None:
        self._exp_dir = exp_dir
        self._names = names
        self._inventory = inventory

    def source(self) -> Path:
        return self._exp_dir / self._names.source

    def vref(self) -> Path:
        return self._exp_dir / self._names.vref

    def predictions(self) -> Path:
        return self._exp_dir / self._names.predictions

    def predictions_detokenized(self) -> Path:
        return self._exp_dir / self._names.predictions_detokenized

    def confidences(self) -> Path:
        return self._exp_dir / (self._names.predictions + CONFIDENCE_SUFFIX)

    def verse_scores(self) -> Path:
        return self._exp_dir / (self._names.predictions_detokenized + VERSE_SCORES_SUFFIX)

    def linregress(self) -> Path:
        return self._exp_dir / self._names.linregress

    def source_iso(self) -> str:
        return self._names.src_iso

    def target_iso(self) -> str:
        return self._names.trg_iso

    def draft_index(self) -> int:
        return self._names.draft_index

    def references(self, ref_projects: Set[str]) -> ReferenceFiles:
        paths = list(self._exp_dir.glob(self._names.references_pattern))
        if len(paths) <= 1:
            return ReferenceFiles(paths, select_random_line=False)
        if len(ref_projects) == 0:
            # With no reference projects named, one reference is built by drawing each verse from a
            # random one of the training references.
            return ReferenceFiles(
                [p for p in paths if self._inventory.is_train_reference(p)], select_random_line=True
            )
        return ReferenceFiles(
            [p for p in paths if self._inventory.references_one_of(ref_projects, p)], select_random_line=False
        )


class ScoreFiles:
    """The files a test run reads and writes in an experiment directory."""

    def __init__(self, exp_dir: Path, inventory: CorpusInventory) -> None:
        self._exp_dir = exp_dir
        self._inventory = inventory

    def has_test_data(self) -> bool:
        return any(self._exp_dir.glob("test*.src.txt"))

    def scores(self, step: int, books: Dict[int, List[int]], ref_projects: Set[str]) -> Path:
        name = f"scores-{self._step_suffix(step, books)}"
        if len(ref_projects) > 0:
            name += "-" + "_".join(sorted(ref_projects))
        return self._exp_dir / f"{name}.csv"

    def test_sets(
        self,
        step: int,
        books: Dict[int, List[int]],
        produce_multiple_translations: bool = False,
        num_drafts: int = 1,
    ) -> List[TestSetFiles]:
        prefixes = self._test_set_prefixes()
        if not produce_multiple_translations:
            return [self._test_set(prefix, step, books, draft=None, draft_index=1) for prefix in prefixes]

        return [
            self._test_set(prefix, step, books, draft=draft, draft_index=draft)
            for draft in range(1, num_drafts + 1)
            for prefix in prefixes
        ]

    def _test_set_prefixes(self) -> List[str]:
        if (self._exp_dir / f"{SINGLE_TEST_SET_PREFIX}.src.txt").is_file():
            return [SINGLE_TEST_SET_PREFIX]
        prefixes: List[str] = []
        for src_iso in sorted(self._inventory.test_source_isos()):
            for trg_iso in sorted(self._inventory.test_target_isos()):
                if src_iso == trg_iso:
                    continue
                prefix = f"test.{src_iso}.{trg_iso}"
                if (self._exp_dir / f"{prefix}.src.txt").is_file():
                    prefixes.append(prefix)
        return prefixes

    def _test_set(
        self, prefix: str, step: int, books: Dict[int, List[int]], draft: Optional[int], draft_index: int
    ) -> TestSetFiles:
        split_by_pair = prefix != SINGLE_TEST_SET_PREFIX
        src_iso, trg_iso = self._isos_of(prefix, split_by_pair)
        suffix = self._step_suffix(step, books)
        draft_part = "" if draft is None else f".{draft}"
        predictions_prefix = f"{prefix}.trg-predictions"
        return TestSetFiles(
            self._exp_dir,
            TestSetNames(
                source=f"{prefix}.src.txt",
                vref=f"{prefix}.vref.txt",
                predictions=f"{predictions_prefix}.txt{draft_part}.{suffix}",
                predictions_detokenized=f"{predictions_prefix}.detok.txt{draft_part}.{suffix}",
                references_pattern=f"{prefix}.trg.detok*.txt",
                linregress=self._linregress_name(
                    step, split_by_pair, src_iso, trg_iso, draft is not None, draft_index
                ),
                src_iso=src_iso,
                trg_iso=trg_iso,
                draft_index=draft_index,
            ),
            self._inventory,
        )

    def _isos_of(self, prefix: str, split_by_pair: bool) -> Tuple[str, str]:
        if not split_by_pair:
            return self._inventory.default_test_source_iso(), self._inventory.default_test_target_iso()
        parts = prefix.split(".")
        return parts[1], parts[2]

    def _linregress_name(
        self,
        step: int,
        split_by_pair: bool,
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool,
        draft_index: int,
    ) -> str:
        parts = [LINREGRESS_PREFIX]
        if split_by_pair:
            parts.extend([src_iso, trg_iso])
        parts.append(self._step_token(step))
        if produce_multiple_translations:
            parts.append(str(draft_index))
        return ".".join(parts) + ".json"

    def _step_suffix(self, step: int, books: Dict[int, List[int]]) -> str:
        book_ids = "_".join(book_number_to_id(num) for num in sorted(books.keys()))
        return f"{book_ids}-{self._step_token(step)}" if len(book_ids) > 0 else self._step_token(step)

    def _step_token(self, step: int) -> str:
        return "avg" if step == -1 else str(step)
