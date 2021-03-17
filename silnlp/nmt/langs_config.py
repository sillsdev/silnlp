import itertools
import os
import random
from glob import glob
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
    get_corpus_path,
    get_scripture_parallel_corpus,
    get_terms,
    get_terms_glosses_path,
    get_terms_list,
    get_terms_renderings_path,
    include_books,
    split_parallel_corpus,
    write_corpus,
)
from ..common.environment import PT_PREPROCESSED_DIR
from ..common.utils import merge_dict
from .config import Config, DataFileType
from .utils import decode_sp_lines, encode_sp, encode_sp_lines, parse_data_file_path


class LangsDataFile:
    def __init__(self, path: str, type: DataFileType):
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
                if (src_rendering, trg_rendering) not in data:
                    data.add((src_rendering, trg_rendering))
    return pd.DataFrame(data, columns=["source", "target"])


def get_terms_data_frame(terms: Dict[str, Term], cats: Optional[Set[str]]) -> pd.DataFrame:
    data: Set[Tuple[str, str]] = set()
    for term in terms.values():
        if cats is not None and term.cat not in cats:
            continue
        for rendering in term.renderings:
            for gloss in term.glosses:
                if (rendering, gloss) not in data:
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
                project_path = get_corpus_path(iso, project)
                data_files.append(
                    LangsDataFile(project_path, DataFileType.TRAIN | DataFileType.TEST | DataFileType.VAL)
                )
        else:
            iso = lang["iso"]
            train_projects = parse_projects(lang.get("train"))
            test_projects = parse_projects(lang.get("test"), default=train_projects)
            val_projects = parse_projects(lang.get("val"), default=train_projects)
            for project in train_projects | test_projects | val_projects:
                file_path = get_corpus_path(iso, project)
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


def get_terms_files(files: List[LangsDataFile], other_isos: Set[str]) -> Tuple[List[LangsDataFile], Set[str]]:
    terms_files: List[LangsDataFile] = []
    glosses_file_paths: Set[str] = set()
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
            if os.path.isfile(glosses_path):
                glosses_file_paths.add(glosses_path)

    return (terms_files, glosses_file_paths)


class LangsConfig(Config):
    def __init__(self, exp_dir: str, config: dict) -> None:
        config = merge_dict(
            {
                "data": {
                    "mixed_src": False,
                    "disjoint_test": False,
                    "disjoint_val": False,
                    "score_threshold": 0,
                    "val_size": 250,
                    "term_cats": "PN",
                    "include_term_glosses": True,
                }
            },
            config,
        )
        data_config: dict = config["data"]
        if "test_size" not in data_config:
            data_config["test_size"] = 0 if "test_books" in data_config else 250

        self.src_files = parse_langs(data_config["src_langs"])
        self.trg_files = parse_langs(data_config["trg_langs"])
        src_isos: Set[str] = set(df.iso for df in self.src_files)
        trg_isos: Set[str] = set(df.iso for df in self.trg_files)
        self.src_terms_files, trg_glosses_file_paths = get_terms_files(self.src_files, trg_isos)
        self.trg_terms_files, src_glosses_file_paths = get_terms_files(self.trg_files, src_isos)
        src_file_paths: Set[str] = set(df.path for df in itertools.chain(self.src_files, self.src_terms_files))
        trg_file_paths: Set[str] = set(df.path for df in itertools.chain(self.trg_files, self.trg_terms_files))
        if data_config["include_term_glosses"]:
            if "en" in src_isos:
                src_file_paths.update(src_glosses_file_paths)
            if "en" in trg_isos:
                trg_file_paths.update(trg_glosses_file_paths)
        super().__init__(exp_dir, config, src_isos, trg_isos, src_file_paths, trg_file_paths)

    def is_train_project(self, ref_file_path: str) -> bool:
        trg_iso, trg_project = self._parse_ref_file_path(ref_file_path)
        for df in self.trg_files:
            if df.iso == trg_iso and df.project == trg_project and df.is_train:
                return True
        return False

    def is_ref_project(self, ref_projects: Set[str], ref_file_path: str) -> bool:
        _, trg_project = self._parse_ref_file_path(ref_file_path)
        return trg_project in ref_projects

    def _build_corpora(self, stats: bool) -> None:
        print("Collecting data sets...")
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

        vref_file_path = os.path.join(PT_PREPROCESSED_DIR, "data", "vref.txt")
        corpus_books = get_books(self.data.get("corpus_books", []))
        test_books = get_books(self.data.get("test_books", []))

        src_spp, trg_spp = self.create_sp_processors()

        stats_file: Optional[IO] = None
        try:
            if stats:
                stats_file = open(os.path.join(self.exp_dir, "corpus-stats.csv"), "w", encoding="utf-8")
                stats_file.write("src_project,trg_project,count,align_score,filtered_count,filtered_align_score\n")

            for src_file in self.src_files:
                for trg_file in self.trg_files:
                    if (
                        train_count(self.src_files) > 1 or train_count(self.trg_files) > 1
                    ) and src_file.iso == trg_file.iso:
                        continue

                    corpus = get_scripture_parallel_corpus(vref_file_path, src_file.path, trg_file.path)
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
                            cur_train.to_csv(
                                os.path.join(self.exp_dir, f"{src_file.project}_{trg_file.project}.csv"), index=False
                            )

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
                            src_file.iso, trg_file.iso, trg_file.project, test, pair_test_indices, cur_test,
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
                            print(f"{src_file.project} -> {trg_file.project} stats")
                            print(f"- count: {corpus_len}")
                            print(f"- alignment: {alignment_score:.4f}")
                            print(f"- filtered count: {filtered_count}")
                            print(f"- alignment (filtered): {filtered_alignment_score:.4f}")
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
                                trg_file.iso, src_file.iso, src_file.project, val, pair_val_indices, mirror_cur_val,
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

        term_cats: Optional[Union[str, List[str]]] = self.data["term_cats"]
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
            if self.data["include_term_glosses"]:
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

        print("Writing train data set...")
        if mixed_src:
            train.fillna("", inplace=True)
            src_columns: List[str] = list(filter(lambda c: c.startswith("source"), train.columns))

            def select_random_column(row: Any) -> pd.Series:
                nonempty_src_columns: List[str] = list(filter(lambda c: row[c] != "", src_columns))
                return row[random.choice(nonempty_src_columns)]

            train["source"] = train[src_columns].apply(select_random_column, axis=1)
            train.drop(src_columns, axis=1, inplace=True, errors="ignore")

        if terms is not None:
            train = pd.concat([train, terms], ignore_index=True)

        write_corpus(os.path.join(self.exp_dir, "train.src.txt"), encode_sp_lines(src_spp, train["source"]))
        write_corpus(os.path.join(self.exp_dir, "train.trg.txt"), encode_sp_lines(trg_spp, train["target"]))

        print("Writing validation data set...")
        if len(val) > 0:
            val_src = itertools.chain.from_iterable(map(lambda pair_val: pair_val["source"], val.values()))
            write_corpus(os.path.join(self.exp_dir, "val.src.txt"), encode_sp_lines(src_spp, val_src))
            self._write_val_corpora(trg_spp, val)

        print("Writing test data set...")
        for old_file_path in glob(os.path.join(self.exp_dir, "test.*.txt")):
            os.remove(old_file_path)

        for (src_iso, trg_iso), pair_test in test.items():
            prefix = "test" if len(test) == 1 else f"test.{src_iso}.{trg_iso}"
            write_corpus(os.path.join(self.exp_dir, f"{prefix}.vref.txt"), map(lambda vr: str(vr), pair_test["vref"]))
            write_corpus(os.path.join(self.exp_dir, f"{prefix}.src.txt"), encode_sp_lines(src_spp, pair_test["source"]))

            columns: List[str] = list(filter(lambda c: c.startswith("target"), pair_test.columns))
            for column in columns:
                project = column[len("target_") :]
                trg_suffix = "" if len(columns) == 1 else f".{project}"
                write_corpus(
                    os.path.join(self.exp_dir, f"{prefix}.trg{trg_suffix}.txt"),
                    encode_sp_lines(trg_spp, pair_test[column]),
                )
                write_corpus(
                    os.path.join(self.exp_dir, f"{prefix}.trg.detok{trg_suffix}.txt"),
                    decode_sp_lines(encode_sp_lines(trg_spp, pair_test[column])),
                )
        print("Preprocessing completed")

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
            terms = pd.concat([terms, cur_terms], ignore_index=True)

        self._insert_trg_tag(trg_iso, cur_terms)

        return pd.concat([terms, cur_terms], ignore_index=True)

    def _insert_trg_tag(self, trg_iso: str, sentences: pd.DataFrame) -> None:
        if self.write_trg_tag:
            sentences.loc[:, "source"] = f"<2{trg_iso}> " + sentences.loc[:, "source"]

    def _write_val_corpora(self, trg_spp: sp.SentencePieceProcessor, val: Dict[Tuple[str, str], pd.DataFrame]) -> None:
        ref_files: List[IO] = []
        try:
            for pair_val in val.values():
                columns: List[str] = list(filter(lambda c: c.startswith("target"), pair_val.columns))
                for index in pair_val.index:
                    if self.root["eval"]["multi_ref_eval"]:
                        for ci in range(len(columns)):
                            if len(ref_files) == ci:
                                ref_files.append(
                                    open(os.path.join(self.exp_dir, f"val.trg.txt.{ci}"), "w", encoding="utf-8")
                                )
                            col = columns[ci]
                            ref_files[ci].write(encode_sp(trg_spp, pair_val.loc[index, col].strip()) + "\n")
                    else:
                        if len(ref_files) == 0:
                            ref_files.append(open(os.path.join(self.exp_dir, "val.trg.txt"), "w", encoding="utf-8"))
                        columns_with_data: List[str] = list(
                            filter(lambda c: pair_val.loc[index, c].strip() != "", columns)
                        )
                        col = random.choice(columns_with_data)
                        ref_files[0].write(encode_sp(trg_spp, pair_val.loc[index, col].strip()) + "\n")

        finally:
            for ref_file in ref_files:
                ref_file.close()

    def _parse_ref_file_path(self, ref_file_path: str) -> Tuple[str, str]:
        parts = os.path.basename(ref_file_path).split(".")
        if len(parts) == 5:
            return self.default_trg_iso, parts[3]
        return parts[2], parts[5]
