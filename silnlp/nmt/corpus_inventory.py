from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union

from ..common.environment import SilNlpEnv
from .corpora import BASIC_DATA_PROJECT, CorpusPair, DataFile, IsoPairInfo, get_terms_glosses_file_paths
from .terms import GlossLanguages


class CorpusInventory:
    def __init__(
        self, corpus_pairs: Iterable[CorpusPair], include_glosses: Union[bool, str], environment: SilNlpEnv
    ) -> None:
        self._corpus_pairs = list(corpus_pairs)
        self._include_glosses = include_glosses
        self._gloss_languages = GlossLanguages()
        self._environment = environment

        self._src_isos: Set[str] = set()
        self._trg_isos: Set[str] = set()
        self._val_src_isos: Set[str] = set()
        self._val_trg_isos: Set[str] = set()
        self._test_src_isos: Set[str] = set()
        self._test_trg_isos: Set[str] = set()
        self._src_file_paths: Set[Path] = set()
        self._trg_file_paths: Set[Path] = set()
        self._src_projects: Set[str] = set()
        self._trg_projects: Set[str] = set()
        self._tags: Set[str] = set()
        self._has_scripture_data = False
        self._iso_pairs: Dict[Tuple[str, str], IsoPairInfo] = {}

        for corpus_pair in self._corpus_pairs:
            self._collect(corpus_pair)

    def has_scripture_data(self) -> bool:
        return self._has_scripture_data

    def has_validation_split(self) -> bool:
        return any(
            pair.is_val and (pair.size if pair.val_size is None else pair.val_size) > 0 for pair in self._corpus_pairs
        )

    def spans_multiple_test_iso_pairs(self) -> bool:
        return sum(1 for iso_pair in self._iso_pairs.values() if iso_pair.has_test_data) > 1

    def has_multiple_test_projects(self, src_iso: str, trg_iso: str) -> bool:
        return self._iso_pairs[(src_iso, trg_iso)].has_multiple_test_projects

    def test_projects(self, src_iso: str, trg_iso: str) -> Set[str]:
        iso_pair = self._iso_pairs[(src_iso, trg_iso)]
        test_projects = iso_pair.test_projects.copy()
        if iso_pair.has_basic_test_data:
            test_projects.add(BASIC_DATA_PROJECT)
        return test_projects

    def validation_project_count(self, src_iso: str, trg_iso: str) -> int:
        return len(self._iso_pairs[(src_iso, trg_iso)].val_projects)

    def is_train_project(self, iso: str, project: str) -> bool:
        for pair in self._corpus_pairs:
            if not pair.is_train:
                continue
            for data_file in pair.src_files + pair.trg_files:
                if data_file.iso == iso and data_file.project == project:
                    return True
        return False

    def is_train_reference(self, prediction_file_path: Path) -> bool:
        return self.is_train_project(*self._reference_of(prediction_file_path))

    def references_one_of(self, projects: Set[str], prediction_file_path: Path) -> bool:
        _, project = self._reference_of(prediction_file_path)
        return project in projects

    def missing_input_files(self) -> List[Path]:
        return sorted(path for path in self._src_file_paths | self._trg_file_paths if not path.is_file())

    def source_isos(self) -> Set[str]:
        return self._src_isos

    def target_isos(self) -> Set[str]:
        return self._trg_isos

    def test_source_isos(self) -> Set[str]:
        return self._test_src_isos

    def test_target_isos(self) -> Set[str]:
        return self._test_trg_isos

    def source_projects(self) -> Set[str]:
        return self._src_projects

    def target_projects(self) -> Set[str]:
        return self._trg_projects

    def source_file_paths(self) -> Set[Path]:
        return self._src_file_paths

    def target_file_paths(self) -> Set[Path]:
        return self._trg_file_paths

    def tags(self) -> Set[str]:
        return self._tags

    def default_test_source_iso(self) -> str:
        return self._first(self._test_src_isos)

    def default_test_target_iso(self) -> str:
        return self._first(self._test_trg_isos)

    def default_validation_source_iso(self) -> str:
        return self._first(self._val_src_isos)

    def default_validation_target_iso(self) -> str:
        return self._first(self._val_trg_isos)

    def _reference_of(self, prediction_file_path: Path) -> Tuple[str, str]:
        parts = prediction_file_path.name.split(".")
        # A name without the iso pair in it belongs to the experiment's only test language pair.
        if len(parts) == 5:
            return self.default_test_target_iso(), parts[3]
        return parts[2], parts[5]

    def _collect(self, corpus_pair: CorpusPair) -> None:
        pair_src_isos = {sf.iso for sf in corpus_pair.src_files}
        pair_trg_isos = {tf.iso for tf in corpus_pair.trg_files}
        self._src_isos.update(pair_src_isos)
        self._trg_isos.update(pair_trg_isos)
        if corpus_pair.is_val:
            self._val_src_isos.update(pair_src_isos)
            self._val_trg_isos.update(pair_trg_isos)
        if corpus_pair.is_test:
            self._test_src_isos.update(pair_src_isos)
            self._test_trg_isos.update(pair_trg_isos)
        self._src_file_paths.update(sf.path for sf in corpus_pair.src_files)
        self._trg_file_paths.update(tf.path for tf in corpus_pair.trg_files)
        if corpus_pair.is_scripture:
            self._collect_scripture(corpus_pair, pair_src_isos, pair_trg_isos)
        self._tags.update(f"<{tag}>" for tag in corpus_pair.tags)
        self._collect_iso_pairs(corpus_pair)

    def _collect_scripture(self, corpus_pair: CorpusPair, pair_src_isos: Set[str], pair_trg_isos: Set[str]) -> None:
        self._has_scripture_data = True
        self._src_file_paths.update(sf.path for sf in corpus_pair.src_terms_files)
        self._trg_file_paths.update(tf.path for tf in corpus_pair.trg_terms_files)
        self._src_projects.update(sf.project for sf in corpus_pair.src_files)
        self._trg_projects.update(tf.project for tf in corpus_pair.trg_files)
        if not self._include_glosses:
            return
        if self._gloss_languages.any_in(pair_src_isos) or self._gloss_languages.includes(self._include_glosses):
            self._src_file_paths.update(self._glosses_of(corpus_pair.src_terms_files))
        if self._gloss_languages.any_in(pair_trg_isos):
            self._trg_file_paths.update(self._glosses_of(corpus_pair.trg_terms_files))

    def _collect_iso_pairs(self, corpus_pair: CorpusPair) -> None:
        for src_file in corpus_pair.src_files:
            for trg_file in corpus_pair.trg_files:
                iso_pair = self._iso_pairs.setdefault((src_file.iso, trg_file.iso), IsoPairInfo())
                if corpus_pair.is_scripture:
                    if corpus_pair.is_test:
                        iso_pair.test_projects.add(trg_file.project)
                    if corpus_pair.is_val:
                        iso_pair.val_projects.add(trg_file.project)
                elif corpus_pair.is_test:
                    iso_pair.has_basic_test_data = True

    def _glosses_of(self, terms_files: List[DataFile]) -> Set[Path]:
        return get_terms_glosses_file_paths(terms_files, environment=self._environment)

    def _first(self, isos: Set[str]) -> str:
        if len(isos) == 0:
            return ""
        return next(iter(isos))
