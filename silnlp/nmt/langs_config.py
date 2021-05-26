import itertools
import random
import logging
from pathlib import Path
from statistics import mean
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas as pd
import sentencepiece as sp

from ..alignment.utils import add_alignment_scores
from ..common.canon import get_books
from ..common.corpus import (
    Term,
    exclude_books,
    filter_parallel_corpus,
    get_scripture_parallel_corpus,
    get_scripture_path,
    get_terms,
    get_terms_glosses_path,
    get_terms_list,
    get_terms_renderings_path,
    include_books,
    load_corpus,
    split_parallel_corpus,
    write_corpus,
)
from ..common.environment import MT_SCRIPTURE_DIR
from ..common.utils import get_mt_exp_dir, merge_dict
from ..common.verse_ref import VerseRef
from .config import Config, DataFileType
from .utils import decode_sp_lines, encode_sp, encode_sp_lines

LOGGER = logging.getLogger(__name__)


def parse_data_file_path(data_file_path: Path) -> Tuple[str, str]:
    file_name = data_file_path.stem
    parts = file_name.split("-")
    return (parts[0], parts[1])


class LangsDataFile:
    def __init__(self, path: Path, type: DataFileType):
        self.path = path
        self.type = type
        self.iso, self.project = parse_data_file_path(path)

    @property
    def is_train(self):
        return (self.type & DataFileType.TRAIN) == DataFileType.TRAIN

    @property
    def is_test(self):
        return (self.type & DataFileType.TEST) == DataFileType.TEST

    @property
    def is_val(self):
        return (self.type & DataFileType.VAL) == DataFileType.VAL


def train_count(data_files: List[LangsDataFile]) -> int:
    return len([df for df in data_files if df.is_train])


def get_terms_corpus(src_terms: Dict[str, Term], trg_terms: Dict[str, Term], cats: Optional[Set[str]]) -> pd.DataFrame:
    data: Set[Tuple[str, str]] = set()
    for src_term in src_terms.values():
        if cats is not None and src_term.cat not in cats:
            continue

        trg_term = trg_terms.get(src_term.id)
        if trg_term is None:
            continue

        for src_rendering in src_term.renderings:
            for trg_rendering in trg_term.renderings:
                data.add((src_rendering, trg_rendering))
    return pd.DataFrame(data, columns=["source", "target"])


def get_terms_data_frame(terms: Dict[str, Term], cats: Optional[Set[str]]) -> pd.DataFrame:
    data: Set[Tuple[str, str]] = set()
    for term in terms.values():
        if cats is not None and term.cat not in cats:
            continue
        for rendering in term.renderings:
            for gloss in term.glosses:
                data.add((rendering, gloss))
    return pd.DataFrame(data, columns=["rendering", "gloss"])


def parse_projects(projects_value: Optional[Union[str, List[str]]], default: Set[str] = set()) -> Set[str]:
    if projects_value is None:
        return default
    if isinstance(projects_value, str):
        return set(map(lambda p: p.strip(), projects_value.split(",")))
    return set(projects_value)


def parse_langs(langs: Iterable[Union[str, dict]]) -> List[LangsDataFile]:
    data_files: List[LangsDataFile] = []
    for lang in langs:
        if isinstance(lang, str):
            index = lang.find("-")
            if index == -1:
                raise RuntimeError("A language project is not fully specified.")
            iso = lang[:index]
            projects_str = lang[index + 1 :]
            for project in projects_str.split(","):
                project = project.strip()
                project_path = get_scripture_path(iso, project)
                data_files.append(
                    LangsDataFile(project_path, DataFileType.TRAIN | DataFileType.TEST | DataFileType.VAL)
                )
        else:
            iso = lang["iso"]
            train_projects = parse_projects(lang.get("train"))
            test_projects = parse_projects(lang.get("test"), default=train_projects)
            val_projects = parse_projects(lang.get("val"), default=train_projects)
            for project in train_projects | test_projects | val_projects:
                file_path = get_scripture_path(iso, project)
                file_type = DataFileType.NONE
                if project in train_projects:
                    file_type |= DataFileType.TRAIN
                if project in test_projects:
                    file_type |= DataFileType.TEST
                if project in val_projects:
                    file_type |= DataFileType.VAL
                data_files.append(LangsDataFile(file_path, file_type))

    data_files.sort(key=lambda df: df.path)
    return data_files


def get_terms_files(files: List[LangsDataFile], other_isos: Set[str]) -> Tuple[List[LangsDataFile], Set[Path]]:
    terms_files: List[LangsDataFile] = []
    glosses_file_paths: Set[Path] = set()
    for file in files:
        if not file.is_train:
            continue
        terms_path = get_terms_renderings_path(file.iso, file.project)
        if terms_path is None:
            continue

        terms_files.append(LangsDataFile(terms_path, DataFileType.TRAIN))
        if "en" in other_isos:
            list_name = get_terms_list(terms_path)
            glosses_path = get_terms_glosses_path(list_name)
            if glosses_path.is_file():
                glosses_file_paths.add(glosses_path)

    return (terms_files, glosses_file_paths)


class LangsConfig(Config):
    def __init__(self, exp_dir: Path, config: dict) -> None:
        config = merge_dict(
            {
                "data": {
                    "mixed_src": False,
                    "disjoint_test": False,
                    "disjoint_val": False,
                    "score_threshold": 0,
                    "val_size": 250,
                    "terms": {
                        "train": True,
                        "dictionary": False,
                        "categories": "PN",
                        "include_glosses": True,
                    },
                }
            },
            config,
        )

        data_config: dict = config["data"]
        if "test_size" not in data_config:
            data_config["test_size"] = 0 if "test_books" in data_config else 250
        if data_config["terms"]["dictionary"]:
            data_config["source_dictionary"] = str(exp_dir / "dict.src.txt")
            data_config["target_dictionary"] = str(exp_dir / "dict.trg.txt")

        self.src_files = parse_langs(data_config["src_langs"])
        self.trg_files = parse_langs(data_config["trg_langs"])
        src_isos: Set[str] = set(df.iso for df in self.src_files)
        trg_isos: Set[str] = set(df.iso for df in self.trg_files)
        self.src_projects: Set[str] = set(df.project for df in self.src_files)
        self.trg_projects: Set[str] = set(df.project for df in self.trg_files)
        self.src_terms_files, trg_glosses_file_paths = get_terms_files(self.src_files, trg_isos)
        self.trg_terms_files, src_glosses_file_paths = get_terms_files(self.trg_files, src_isos)
        src_file_paths: Set[Path] = set(df.path for df in itertools.chain(self.src_files, self.src_terms_files))
        trg_file_paths: Set[Path] = set(df.path for df in itertools.chain(self.trg_files, self.trg_terms_files))
        if data_config["terms"]["include_glosses"]:
            if "en" in src_isos:
                src_file_paths.update(src_glosses_file_paths)
            if "en" in trg_isos:
                trg_file_paths.update(trg_glosses_file_paths)
        src_tags: Set[str] = set()
        super().__init__(exp_dir, config, src_isos, trg_isos, src_file_paths, trg_file_paths, src_tags)

    def is_train_project(self, ref_file_path: Path) -> bool:
        trg_iso, trg_project = self._parse_ref_file_path(ref_file_path)
        for df in self.trg_files:
            if df.iso == trg_iso and df.project == trg_project and df.is_train:
                return True
        return False

    def is_ref_project(self, ref_projects: Set[str], ref_file_path: Path) -> bool:
        _, trg_project = self._parse_ref_file_path(ref_file_path)
        return trg_project in ref_projects

    def _build_corpora(
        self, src_spp: Optional[sp.SentencePieceProcessor], trg_spp: Optional[sp.SentencePieceProcessor], stats: bool
    ) -> None:
        LOGGER.info("Collecting data sets...")
        test_size: int = self.data["test_size"]
        val_size: int = self.data["val_size"]
        disjoint_test: bool = self.data["disjoint_test"]
        disjoint_val: bool = self.data["disjoint_val"]
        score_threshold: float = self.data["score_threshold"]
        mixed_src: bool = self.data["mixed_src"] and train_count(self.src_files) > 1

        test_indices: Optional[Set[int]] = None
        val_indices: Optional[Set[int]] = None

        train: Optional[pd.DataFrame] = None
        val: Dict[Tuple[str, str], pd.DataFrame] = {}
        test: Dict[Tuple[str, str], pd.DataFrame] = {}
        pair_val_indices: Dict[Tuple[str, str], Set[int]] = {}
        pair_test_indices: Dict[Tuple[str, str], Set[int]] = {}
        terms: Optional[pd.DataFrame] = None

        self._populate_pair_test_indices(pair_test_indices)

        corpus_books = get_books(self.data.get("corpus_books", []))
        test_books = get_books(self.data.get("test_books", []))

        stats_file: Optional[IO] = None
        try:
            if stats:
                stats_file = open(self.exp_dir / "corpus-stats.csv", "w", encoding="utf-8")
                stats_file.write("src_project,trg_project,count,align_score,filtered_count,filtered_align_score\n")

            for src_file in self.src_files:
                for trg_file in self.trg_files:
                    if (
                        train_count(self.src_files) > 1 or train_count(self.trg_files) > 1
                    ) and src_file.iso == trg_file.iso:
                        continue

                    corpus = get_scripture_parallel_corpus(src_file.path, trg_file.path)
                    if len(corpus_books) > 0:
                        cur_train = include_books(corpus, corpus_books)
                        if len(corpus_books.intersection(test_books)) > 0:
                            cur_train = exclude_books(cur_train, test_books)
                    elif len(test_books) > 0:
                        cur_train = exclude_books(corpus, test_books)
                    else:
                        cur_train = corpus

                    corpus_len = len(cur_train)
                    if trg_file.is_train and (stats_file is not None or score_threshold > 0):
                        add_alignment_scores(cur_train)
                        if stats_file is not None:
                            cur_train.to_csv(self.exp_dir / f"{src_file.project}_{trg_file.project}.csv", index=False)

                    if trg_file.is_test:
                        if disjoint_test and test_indices is None:
                            indices: Set[int] = set(cur_train.index)
                            if disjoint_val and val_indices is not None:
                                indices.difference_update(val_indices)
                            test_indices = set(random.sample(indices, min(test_size, len(indices))))

                        if len(test_books) > 0:
                            cur_test = include_books(corpus, test_books)
                            if test_size > 0:
                                _, cur_test = split_parallel_corpus(
                                    cur_test,
                                    test_size,
                                    pair_test_indices.get((src_file.iso, trg_file.iso), test_indices),
                                )
                        else:
                            cur_train, cur_test = split_parallel_corpus(
                                cur_train, test_size, pair_test_indices.get((src_file.iso, trg_file.iso), test_indices)
                            )

                        cur_test.drop("score", axis=1, inplace=True, errors="ignore")
                        self._add_to_eval_dataset(
                            src_file.iso,
                            trg_file.iso,
                            trg_file.project,
                            test,
                            pair_test_indices,
                            cur_test,
                        )

                    if trg_file.is_train:
                        alignment_score = mean(cur_train["score"]) if stats_file is not None else 0

                        filtered_count = 0
                        filtered_alignment_score = alignment_score
                        if score_threshold > 0:
                            unfiltered_len = len(cur_train)
                            cur_train = filter_parallel_corpus(cur_train, score_threshold)
                            filtered_count = unfiltered_len - len(cur_train)
                            filtered_alignment_score = mean(cur_train["score"]) if stats_file is not None else 0

                        if stats_file is not None:
                            LOGGER.info(f"{src_file.project} -> {trg_file.project} stats")
                            LOGGER.info(f"- count: {corpus_len}")
                            LOGGER.info(f"- alignment: {alignment_score:.4f}")
                            LOGGER.info(f"- filtered count: {filtered_count}")
                            LOGGER.info(f"- alignment (filtered): {filtered_alignment_score:.4f}")
                            stats_file.write(
                                f"{src_file.project},{trg_file.project},{corpus_len},{alignment_score:.4f},"
                                f"{filtered_count},{filtered_alignment_score:.4f}\n"
                            )
                        cur_train.drop("score", axis=1, inplace=True, errors="ignore")

                    if trg_file.is_val:
                        if disjoint_val and val_indices is None:
                            indices = set(cur_train.index)
                            if disjoint_test and test_indices is not None:
                                indices.difference_update(test_indices)
                            val_indices = set(random.sample(indices, min(val_size, len(indices))))

                        cur_train, cur_val = split_parallel_corpus(
                            cur_train, val_size, pair_val_indices.get((src_file.iso, trg_file.iso), val_indices)
                        )

                        if self.mirror:
                            mirror_cur_val = cur_val.rename(columns={"source": "target", "target": "source"})
                            self._add_to_eval_dataset(
                                trg_file.iso,
                                src_file.iso,
                                src_file.project,
                                val,
                                pair_val_indices,
                                mirror_cur_val,
                            )

                        self._add_to_eval_dataset(
                            src_file.iso, trg_file.iso, trg_file.project, val, pair_val_indices, cur_val
                        )

                    if trg_file.is_train:
                        if self.mirror:
                            mirror_cur_train = cur_train.rename(columns={"source": "target", "target": "source"})
                            train = self._add_to_train_dataset(
                                src_file.iso, trg_file.project, src_file.project, mixed_src, train, mirror_cur_train
                            )

                        train = self._add_to_train_dataset(
                            trg_file.iso, src_file.project, trg_file.project, mixed_src, train, cur_train
                        )

        finally:
            if stats_file is not None:
                stats_file.close()

        terms_config = self.data["terms"]
        if terms_config["train"] or terms_config["dictionary"]:
            term_cats: Optional[Union[str, List[str]]] = terms_config["categories"]
            if isinstance(term_cats, str):
                term_cats = [cat.strip() for cat in term_cats.split(",")]
            if term_cats is None or len(term_cats) > 0:
                term_cats_set: Optional[Set[str]] = None if term_cats is None else set(term_cats)
                all_src_terms = [
                    (src_terms_file, get_terms(src_terms_file.path)) for src_terms_file in self.src_terms_files
                ]
                all_trg_terms = [
                    (trg_terms_file, get_terms(trg_terms_file.path)) for trg_terms_file in self.trg_terms_files
                ]
                for src_terms_file, src_terms in all_src_terms:
                    for trg_terms_file, trg_terms in all_trg_terms:
                        if src_terms_file.iso == trg_terms_file.iso:
                            continue
                        cur_terms = get_terms_corpus(src_terms, trg_terms, term_cats_set)
                        terms = self._add_to_terms_dataset(src_terms_file.iso, trg_terms_file.iso, terms, cur_terms)
                if terms_config["include_glosses"]:
                    if "en" in self.trg_isos:
                        for src_terms_file, src_terms in all_src_terms:
                            cur_terms = get_terms_data_frame(src_terms, term_cats_set)
                            cur_terms = cur_terms.rename(columns={"rendering": "source", "gloss": "target"})
                            terms = self._add_to_terms_dataset(src_terms_file.iso, "en", terms, cur_terms)
                    if "en" in self.src_isos:
                        for trg_terms_file, trg_terms in all_trg_terms:
                            cur_terms = get_terms_data_frame(trg_terms, term_cats_set)
                            cur_terms = cur_terms.rename(columns={"rendering": "target", "gloss": "source"})
                            terms = self._add_to_terms_dataset("en", trg_terms_file.iso, terms, cur_terms)

        if train is None:
            return

        LOGGER.info("Writing train data set...")
        if mixed_src:
            train.fillna("", inplace=True)
            src_columns: List[str] = list(filter(lambda c: c.startswith("source"), train.columns))

            def select_random_column(row: Any) -> pd.Series:
                nonempty_src_columns: List[str] = list(filter(lambda c: row[c] != "", src_columns))
                return row[random.choice(nonempty_src_columns)]

            train["source"] = train[src_columns].apply(select_random_column, axis=1)
            train.drop(src_columns, axis=1, inplace=True, errors="ignore")

        write_corpus(self.exp_dir / "train.src.txt", encode_sp_lines(src_spp, train["source"]))
        write_corpus(self.exp_dir / "train.trg.txt", encode_sp_lines(trg_spp, train["target"]))
        write_corpus(self.exp_dir / "train.vref.txt", (str(vr) for vr in train["vref"]))

        if terms is not None:
            self._write_terms(src_spp, trg_spp, terms)

        LOGGER.info("Writing validation data set...")
        if len(val) > 0:
            val_src = itertools.chain.from_iterable(map(lambda pair_val: pair_val["source"], val.values()))
            write_corpus(self.exp_dir / "val.src.txt", encode_sp_lines(src_spp, val_src))
            self._write_val_corpora(trg_spp, val)
            val_vref = itertools.chain.from_iterable(map(lambda pair_val: pair_val["vref"], val.values()))
            write_corpus(self.exp_dir / "val.vref.txt", map(lambda vr: str(vr), val_vref))

        LOGGER.info("Writing test data set...")
        for old_file_path in self.exp_dir.glob("test.*.txt"):
            old_file_path.unlink()

        for (src_iso, trg_iso), pair_test in test.items():
            prefix = "test" if len(test) == 1 else f"test.{src_iso}.{trg_iso}"
            write_corpus(self.exp_dir / f"{prefix}.vref.txt", map(lambda vr: str(vr), pair_test["vref"]))
            write_corpus(self.exp_dir / f"{prefix}.src.txt", encode_sp_lines(src_spp, pair_test["source"]))

            columns: List[str] = [c for c in pair_test.columns if c.startswith("target")]
            for column in columns:
                project = column[len("target_") :]
                trg_suffix = "" if len(columns) == 1 else f".{project}"
                write_corpus(
                    self.exp_dir / f"{prefix}.trg{trg_suffix}.txt",
                    encode_sp_lines(trg_spp, pair_test[column]),
                )
                if trg_spp is not None:
                    write_corpus(
                        self.exp_dir / f"{prefix}.trg.detok{trg_suffix}.txt",
                        decode_sp_lines(encode_sp_lines(trg_spp, pair_test[column])),
                    )

    def _add_to_train_dataset(
        self,
        trg_iso: str,
        src_project: str,
        trg_project: str,
        mixed_src: bool,
        train: pd.DataFrame,
        cur_train: pd.DataFrame,
    ) -> pd.DataFrame:
        self._insert_trg_tag(trg_iso, cur_train)
        if mixed_src:
            cur_train.rename(columns={"source": f"source_{src_project}"}, inplace=True)
            cur_train.set_index(
                pd.MultiIndex.from_tuples(
                    map(lambda i: (trg_project, i), cur_train.index), names=["trg_project", "index"]
                ),
                inplace=True,
            )
            train = cur_train if train is None else train.combine_first(cur_train)
        else:
            train = pd.concat([train, cur_train], ignore_index=True)
        return train

    def _add_to_eval_dataset(
        self,
        src_iso: str,
        trg_iso: str,
        trg_project: str,
        dataset: Dict[Tuple[str, str], pd.DataFrame],
        pair_indices: Dict[Tuple[str, str], Set[int]],
        new_data: pd.DataFrame,
    ) -> None:
        if len(new_data) == 0:
            return

        self._insert_trg_tag(trg_iso, new_data)

        pair_data = dataset.get((src_iso, trg_iso))

        if (src_iso, trg_iso) not in pair_indices:
            pair_indices[(src_iso, trg_iso)] = set(new_data.index)

        new_data.rename(columns={"target": f"target_{trg_project}"}, inplace=True)
        if pair_data is None:
            pair_data = new_data
        else:
            pair_data = pair_data.combine_first(new_data)
            pair_data.fillna("", inplace=True)

        dataset[(src_iso, trg_iso)] = pair_data

    def _add_to_terms_dataset(
        self, src_iso: str, trg_iso: str, terms: pd.DataFrame, cur_terms: pd.DataFrame
    ) -> pd.DataFrame:
        if self.mirror:
            mirror_cur_terms = cur_terms.rename(columns={"source": "target", "target": "source"})
            self._insert_trg_tag(src_iso, mirror_cur_terms)
            terms = pd.concat([terms, mirror_cur_terms], ignore_index=True)

        self._insert_trg_tag(trg_iso, cur_terms)

        return pd.concat([terms, cur_terms], ignore_index=True)

    def _insert_trg_tag(self, trg_iso: str, sentences: pd.DataFrame) -> None:
        if self.write_trg_tag:
            sentences.loc[:, "source"] = f"<2{trg_iso}> " + sentences.loc[:, "source"]

    def _write_val_corpora(
        self, trg_spp: Optional[sp.SentencePieceProcessor], val: Dict[Tuple[str, str], pd.DataFrame]
    ) -> None:
        ref_files: List[IO] = []
        try:
            for pair_val in val.values():
                columns: List[str] = list(filter(lambda c: c.startswith("target"), pair_val.columns))
                for index in pair_val.index:
                    if self.root["eval"]["multi_ref_eval"]:
                        for ci in range(len(columns)):
                            if len(ref_files) == ci:
                                ref_files.append(open(self.exp_dir / f"val.trg.txt.{ci}", "w", encoding="utf-8"))
                            col = columns[ci]
                            ref_files[ci].write(encode_sp(trg_spp, pair_val.loc[index, col].strip()) + "\n")
                    else:
                        if len(ref_files) == 0:
                            ref_files.append(open(self.exp_dir / "val.trg.txt", "w", encoding="utf-8"))
                        columns_with_data: List[str] = list(
                            filter(lambda c: pair_val.loc[index, c].strip() != "", columns)
                        )
                        col = random.choice(columns_with_data)
                        ref_files[0].write(encode_sp(trg_spp, pair_val.loc[index, col].strip()) + "\n")

        finally:
            for ref_file in ref_files:
                ref_file.close()

    def _parse_ref_file_path(self, ref_file_path: Path) -> Tuple[str, str]:
        parts = ref_file_path.name.split(".")
        if len(parts) == 5:
            return self.default_trg_iso, parts[3]
        return parts[2], parts[5]

    def _populate_pair_test_indices(self, pair_test_indices: Dict[Tuple[str, str], Set[int]]) -> None:
        exp_name = self.data.get("use_test_set_from")
        if exp_name is None:
            return

        vrefs: Dict[str, int] = {}
        for i, vref_str in enumerate(load_corpus(MT_SCRIPTURE_DIR / "vref.txt")):
            vrefs[vref_str] = i

        exp_dir = get_mt_exp_dir(exp_name)
        for vref_path in exp_dir.glob("test*.vref.txt"):
            stem = vref_path.stem
            test_indices: Set[int] = set()
            if stem == "test.vref":
                pair_test_indices[(self.default_src_iso, self.default_trg_iso)] = test_indices
            else:
                _, src_iso, trg_iso, _ = stem.split(".", maxsplit=4)
                pair_test_indices[(src_iso, trg_iso)] = test_indices

            for vref_str in load_corpus(vref_path):
                vref = VerseRef.from_string(vref_str)
                if vref.has_multiple:
                    vref = vref.simplify()
                test_indices.add(vrefs[str(vref)])

    def _write_terms(
        self,
        src_spp: Optional[sp.SentencePieceProcessor],
        trg_spp: Optional[sp.SentencePieceProcessor],
        terms: pd.DataFrame,
    ) -> None:
        train_src_file: Optional[IO] = None
        train_trg_file: Optional[IO] = None
        train_vref_file: Optional[IO] = None

        dict_src_file: Optional[IO] = None
        dict_trg_file: Optional[IO] = None
        try:
            terms_config = self.data["terms"]
            if terms_config["train"]:
                train_src_file = open(self.exp_dir / "train.src.txt", "a", encoding="utf-8", newline="\n")
                train_trg_file = open(self.exp_dir / "train.trg.txt", "a", encoding="utf-8", newline="\n")
                train_vref_file = open(self.exp_dir / "train.vref.txt", "a", encoding="utf-8", newline="\n")

            if terms_config["dictionary"]:
                dict_src_file = open(self.exp_dir / "dict.src.txt", "w", encoding="utf-8", newline="\n")
                dict_trg_file = open(self.exp_dir / "dict.trg.txt", "w", encoding="utf-8", newline="\n")

            for _, term in terms.iterrows():
                src_term: str = term["source"]
                trg_term: str = term["target"]
                src_term_variants = [
                    encode_sp(src_spp, src_term, add_dummy_prefix=True),
                    encode_sp(src_spp, src_term, add_dummy_prefix=False),
                ]
                trg_term_variants = [
                    encode_sp(trg_spp, trg_term, add_dummy_prefix=True),
                    encode_sp(trg_spp, trg_term, add_dummy_prefix=False),
                ]

                if train_src_file is not None and train_trg_file is not None and train_vref_file is not None:
                    for stv in src_term_variants:
                        for ttv in trg_term_variants:
                            train_src_file.write(stv + "\n")
                            train_trg_file.write(ttv + "\n")
                            train_vref_file.write("\n")

                if dict_src_file is not None and dict_trg_file is not None:
                    dict_src_file.write("\t".join(src_term_variants) + "\n")
                    dict_trg_file.write("\t".join(trg_term_variants) + "\n")
        finally:
            if train_src_file is not None:
                train_src_file.close()
            if train_trg_file is not None:
                train_trg_file.close()
            if train_vref_file is not None:
                train_vref_file.close()

            if dict_src_file is not None:
                dict_src_file.close()
            if dict_trg_file is not None:
                dict_trg_file.close()
