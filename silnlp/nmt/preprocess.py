import argparse
import itertools
import logging
import os
import random
import shutil
from enum import Enum
from glob import glob
from statistics import mean
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple

logging.basicConfig()

import pandas as pd
import sentencepiece as sp
from opennmt import END_OF_SENTENCE_TOKEN, PADDING_TOKEN, START_OF_SENTENCE_TOKEN
from opennmt.data import Vocab

from ..alignment.utils import add_alignment_scores
from ..common.canon import get_books
from ..common.corpus import (
    exclude_books,
    filter_parallel_corpus,
    get_names_path,
    get_scripture_parallel_corpus,
    include_books,
    load_corpus,
    split_parallel_corpus,
    write_corpus,
)
from ..common.environment import PT_PREPROCESSED_DIR
from ..common.utils import set_seed
from .config import (
    DataFile,
    DataFileType,
    Language,
    create_runner,
    get_git_revision_hash,
    get_mt_root_dir,
    load_config,
    parse_langs,
)
from .utils import decode_sp, decode_sp_lines, encode_sp, encode_sp_lines, get_best_model_dir, get_last_checkpoint


# Different types of parent model checkpoints (last, best, average)
class CheckpointType(Enum):
    LAST = 1
    BEST = 2
    AVERAGE = 3


def convert_vocab(sp_vocab_path: str, onmt_vocab_path: str, tag_langs: Set[str] = None) -> None:
    special_tokens = [PADDING_TOKEN, START_OF_SENTENCE_TOKEN, END_OF_SENTENCE_TOKEN]
    if tag_langs is not None:
        special_tokens.extend(map(lambda l: f"<2{l}>", tag_langs))

    vocab = Vocab(special_tokens)
    with open(sp_vocab_path, "r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            token = line.rstrip("\r\n")
            index = token.rindex("\t")
            token = token[:index]
            if token in ("<unk>", "<s>", "</s>", "<range>"):  # Ignore special tokens
                continue
            vocab.add(token)
    vocab.pad_to_multiple(8)
    vocab.serialize(onmt_vocab_path)


def build_vocab(
    file_paths: Iterable[str],
    vocab_size: int,
    casing: str,
    character_coverage: float,
    model_prefix: str,
    vocab_path: str,
    tag_langs: Set[str] = None,
) -> None:
    joined_file_paths = ",".join(file_paths)

    casing = casing.lower()
    normalization: str
    if casing == "lower":
        normalization = "nmt_nfkc_cf"
    elif casing == "preserve":
        normalization = "nmt_nfkc"
    else:
        raise RuntimeError("Invalid casing was specified in the config.")

    # use custom normalization that does not convert ZWJ and ZWNJ to spaces
    # allows properly handling of scripts like Devanagari
    normalization_path = os.path.join(os.path.dirname(__file__), f"{normalization}.tsv")
    sp_train_params = (
        f"--normalization_rule_tsv={normalization_path} --input={joined_file_paths} --model_prefix={model_prefix}"
        f" --vocab_size={vocab_size} --character_coverage={character_coverage:.4f} --input_sentence_size=1000000"
        " --shuffle_input_sentence=true --control_symbols=<range> --user_defined_symbols=[[BLANK]]"
    )

    sp.SentencePieceTrainer.Train(sp_train_params)

    convert_vocab(f"{model_prefix}.vocab", vocab_path, tag_langs)


def insert_trg_tag(trg_iso: str, sentences: pd.DataFrame) -> None:
    sentences.loc[:, "source"] = f"<2{trg_iso}> " + sentences.loc[:, "source"]


def get_checkpoint_path(model_dir: str, checkpoint_type: CheckpointType) -> Tuple[Optional[str], Optional[int]]:
    if checkpoint_type == CheckpointType.AVERAGE:
        # Get the checkpoint path and step count for the averaged checkpoint
        return get_last_checkpoint(os.path.join(model_dir, "avg"))
    elif checkpoint_type == CheckpointType.BEST:
        # Get the checkpoint path and step count for the best checkpoint
        best_model_dir, step = get_best_model_dir(model_dir)
        return (os.path.join(best_model_dir, "ckpt"), step)
    elif checkpoint_type == CheckpointType.LAST:
        return (None, None)
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {checkpoint_type}")


def update_vocab(
    parent_config: dict, root_dir: str, src_vocab_path: str, trg_vocab_path: str, parent_model_to_use: CheckpointType
) -> None:
    model_dir: str = parent_config["model_dir"]
    checkpoint_path, step = get_checkpoint_path(model_dir, parent_model_to_use)
    parent_runner = create_runner(parent_config)
    parent_runner.update_vocab(os.path.join(root_dir, "parent"), src_vocab_path, trg_vocab_path, checkpoint_path, step)


def is_corpus_in_langs(langs: Set[str], lang_projects: Dict[str, Set[str]], iso: str, project: str) -> bool:
    if iso in langs:
        projects = lang_projects[iso]
        return project in projects
    return False


def create_unshared_vocab(
    root_dir: str,
    data_config: dict,
    parent_root_dir: str,
    parent_data_config: dict,
    parent_use_vocab: bool,
    langs: Dict[str, Language],
    vocab_file_paths: Set[str],
    side: str,
    tag_langs: Set[str] = None,
) -> None:
    prefix = "src" if side == "source" else "trg"
    model_prefix = os.path.join(root_dir, f"{prefix}-sp")
    vocab_path = os.path.join(root_dir, f"{prefix}-onmt.vocab")
    parent_langs = parse_langs(parent_data_config.get(f"{prefix}_langs", []))
    if langs.keys() == parent_langs.keys():
        parent_sp_prefix_path: str
        parent_vocab_path: str
        if parent_data_config["share_vocab"]:
            parent_sp_prefix_path = os.path.join(parent_root_dir, "sp")
            parent_vocab_path = os.path.join(parent_root_dir, "onmt.vocab")
        else:
            parent_sp_prefix_path = os.path.join(parent_root_dir, f"{prefix}-sp")
            parent_vocab_path = os.path.join(parent_root_dir, f"{prefix}-onmt.vocab")

        parent_vocab: Optional[Vocab] = None
        child_tokens: Optional[Set[str]] = None
        if not parent_use_vocab:
            parent_spp = sp.SentencePieceProcessor()
            parent_spp.Load(parent_sp_prefix_path + ".model")

            parent_vocab = Vocab()
            parent_vocab.load(parent_vocab_path)

            child_tokens = set()
            for vocab_file_path in vocab_file_paths:
                for line in encode_sp_lines(parent_spp, load_corpus(vocab_file_path)):
                    child_tokens.update(line.split())
            parent_use_vocab = child_tokens.issubset(parent_vocab.words)

        # all tokens in the child corpora are in the parent vocab, so we can just use the parent vocab
        # or, the user wants to reuse the parent vocab for this child experiment
        if child_tokens is None or parent_vocab is None:
            sp_vocab_path = os.path.join(root_dir, f"{prefix}-sp.vocab")
            onmt_vocab_path = os.path.join(root_dir, f"{prefix}-onmt.vocab")
            shutil.copy2(parent_sp_prefix_path + ".model", os.path.join(root_dir, f"{prefix}-sp.model"))
            shutil.copy2(parent_sp_prefix_path + ".vocab", sp_vocab_path)
            convert_vocab(sp_vocab_path, onmt_vocab_path, tag_langs)
            return
        else:
            onmt_delta_vocab_path = os.path.join(root_dir, f"{prefix}-onmt-delta.vocab")
            vocab_delta = child_tokens.difference(parent_vocab.words)
            with open(onmt_delta_vocab_path, "w", encoding="utf-8") as f:
                [f.write(f"{token}\n") for token in vocab_delta]

    print(f"Building {side} vocabulary...")
    vocab_size: int = data_config.get(f"{prefix}_vocab_size", data_config.get("vocab_size"))
    casing: str = data_config.get(f"{prefix}_casing", data_config.get("casing"))
    character_coverage: float = data_config.get(f"{prefix}_character_coverage", data_config.get("character_coverage"))
    build_vocab(vocab_file_paths, vocab_size, casing, character_coverage, model_prefix, vocab_path, tag_langs)


def add_to_eval_dataset(
    src_iso: str,
    trg_iso: str,
    trg_project: str,
    write_trg_tag: bool,
    dataset: Dict[Tuple[str, str], pd.DataFrame],
    pair_indices: Dict[Tuple[str, str], Set[int]],
    new_data: pd.DataFrame,
) -> None:
    if len(new_data) == 0:
        return

    if write_trg_tag:
        insert_trg_tag(trg_iso, new_data)

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


def add_to_train_dataset(
    src_project: str, trg_project: str, mixed_src: bool, train: pd.DataFrame, cur_train: pd.DataFrame
) -> pd.DataFrame:
    if mixed_src:
        cur_train.rename(columns={"source": f"source_{src_project}"}, inplace=True)
        cur_train.set_index(
            pd.MultiIndex.from_tuples(map(lambda i: (trg_project, i), cur_train.index), names=["trg_project", "index"]),
            inplace=True,
        )
        train = cur_train if train is None else train.combine_first(cur_train)
    else:
        train = pd.concat([train, cur_train], ignore_index=True)
    return train


def write_val_corpora(
    trg_spp: sp.SentencePieceProcessor, multi_ref_eval: bool, val: Dict[Tuple[str, str], pd.DataFrame], root_dir: str
) -> None:
    ref_files: List[IO] = []
    try:
        for pair_val in val.values():
            columns: List[str] = list(filter(lambda c: c.startswith("target"), pair_val.columns))
            for index in pair_val.index:
                if multi_ref_eval:
                    for ci in range(len(columns)):
                        if len(ref_files) == ci:
                            ref_files.append(open(os.path.join(root_dir, f"val.trg.txt.{ci}"), "w", encoding="utf-8"))
                        col = columns[ci]
                        ref_files[ci].write(encode_sp(trg_spp, pair_val.loc[index, col].strip()) + "\n")
                else:
                    if len(ref_files) == 0:
                        ref_files.append(open(os.path.join(root_dir, "val.trg.txt"), "w", encoding="utf-8"))
                    columns_with_data: List[str] = list(filter(lambda c: pair_val.loc[index, c].strip() != "", columns))
                    col = random.choice(columns_with_data)
                    ref_files[0].write(encode_sp(trg_spp, pair_val.loc[index, col].strip()) + "\n")

    finally:
        for ref_file in ref_files:
            ref_file.close()


def get_data_files(langs: Dict[str, Language]) -> List[DataFile]:
    data_files = [df for lang in langs.values() for df in lang.data_files]
    data_files.sort(key=lambda df: df.path)
    return data_files


def train_count(data_files: List[DataFile]) -> int:
    return len([df for df in data_files if df.is_train])


def get_names_files(src_files: List[DataFile], trg_langs: Dict[str, Language]) -> Tuple[List[DataFile], List[DataFile]]:
    src_names_files: List[DataFile] = []
    trg_names_files: List[DataFile] = []
    for src_file in src_files:
        if not src_file.is_train:
            continue
        src_names_path = get_names_path(src_file.iso, src_file.project)
        if os.path.isfile(src_names_path):
            for trg_iso in trg_langs.keys():
                trg_names_path = get_names_path(trg_iso, src_file.project)
                if os.path.isfile(trg_names_path):
                    src_names_files.append(DataFile(src_names_path, DataFileType.TRAIN))
                    trg_names_files.append(DataFile(trg_names_path, DataFileType.TRAIN))
    return (src_names_files, trg_names_files)


def get_names_corpus(src_names_path: str, trg_names_path: str) -> pd.DataFrame:
    src_names: List[str] = []
    trg_names: List[str] = []
    with open(src_names_path, "r", encoding="utf-8") as src_file, open(
        trg_names_path, "r", encoding="utf-8"
    ) as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            src_names.append(src_line)
            trg_names.append(trg_line)

    data = {"source": src_names, "target": trg_names}
    return pd.DataFrame(data)


def preprocess_scripture(
    root_dir: str,
    config: dict,
    args: argparse.Namespace,
    src_files: List[DataFile],
    trg_files: List[DataFile],
    src_names_files: List[DataFile],
    trg_names_files: List[DataFile],
    write_trg_tag: bool,
    src_spp: Optional[sp.SentencePieceProcessor],
    trg_spp: Optional[sp.SentencePieceProcessor],
) -> None:
    print("Collecting data sets...")
    data_config: dict = config["data"]
    test_size: int = data_config["test_size"]
    val_size: int = data_config["val_size"]
    disjoint_test: bool = data_config["disjoint_test"]
    disjoint_val: bool = data_config["disjoint_val"]
    score_threshold: float = data_config["score_threshold"]
    mixed_src: bool = data_config["mixed_src"] and train_count(src_files) > 1
    mirror: bool = data_config["mirror"]
    multi_ref_eval: bool = config["eval"]["multi_ref_eval"]

    test_indices: Optional[Set[int]] = None
    val_indices: Optional[Set[int]] = None

    train: Optional[pd.DataFrame] = None
    val: Dict[Tuple[str, str], pd.DataFrame] = {}
    test: Dict[Tuple[str, str], pd.DataFrame] = {}
    pair_val_indices: Dict[Tuple[str, str], Set[int]] = {}
    pair_test_indices: Dict[Tuple[str, str], Set[int]] = {}
    names: Optional[pd.DataFrame] = None

    vref_file_path = os.path.join(PT_PREPROCESSED_DIR, "data", "vref.txt")
    corpus_books = get_books(data_config.get("corpus_books", []))
    test_books = get_books(data_config.get("test_books", []))

    stats_file: Optional[IO] = None
    try:
        if args.stats:
            stats_file = open(os.path.join(root_dir, "corpus-stats.csv"), "w", encoding="utf-8")
            stats_file.write("src_project,trg_project,count,align_score,filtered_count,filtered_align_score\n")

        for src_file in src_files:
            for trg_file in trg_files:
                if (train_count(src_files) > 1 or train_count(trg_files) > 1) and src_file.iso == trg_file.iso:
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
                            os.path.join(root_dir, f"{src_file.project}_{trg_file.project}.csv"), index=False
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
                                cur_test, test_size, pair_test_indices.get((src_file.iso, trg_file.iso), test_indices)
                            )
                    else:
                        cur_train, cur_test = split_parallel_corpus(
                            cur_train, test_size, pair_test_indices.get((src_file.iso, trg_file.iso), test_indices)
                        )

                    cur_test.drop("score", axis=1, inplace=True, errors="ignore")
                    add_to_eval_dataset(
                        src_file.iso, trg_file.iso, trg_file.project, write_trg_tag, test, pair_test_indices, cur_test
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

                    if mirror:
                        mirror_cur_val = cur_val.rename(columns={"source": "target", "target": "source"})
                        add_to_eval_dataset(
                            trg_file.iso,
                            src_file.iso,
                            src_file.project,
                            write_trg_tag,
                            val,
                            pair_val_indices,
                            mirror_cur_val,
                        )

                    add_to_eval_dataset(
                        src_file.iso, trg_file.iso, trg_file.project, write_trg_tag, val, pair_val_indices, cur_val
                    )

                if trg_file.is_train:
                    if mirror:
                        mirror_cur_train = cur_train.rename(columns={"source": "target", "target": "source"})
                        if write_trg_tag:
                            insert_trg_tag(src_file.iso, mirror_cur_train)
                        train = add_to_train_dataset(
                            trg_file.project, src_file.project, mixed_src, train, mirror_cur_train
                        )

                    if write_trg_tag:
                        insert_trg_tag(trg_file.iso, cur_train)

                    train = add_to_train_dataset(src_file.project, trg_file.project, mixed_src, train, cur_train)

    finally:
        if stats_file is not None:
            stats_file.close()

    for src_names_file, trg_names_file in zip(src_names_files, trg_names_files):
        cur_names = get_names_corpus(src_names_file.path, trg_names_file.path)
        if mirror:
            mirror_cur_names = cur_names.rename(columns={"source": "target", "target": "source"})
            if write_trg_tag:
                insert_trg_tag(src_names_file.iso, mirror_cur_names)
            names = pd.concat([names, cur_names], ignore_index=True)

        if write_trg_tag:
            insert_trg_tag(trg_names_file.iso, cur_names)

        names = pd.concat([names, cur_names], ignore_index=True)

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

    if names is not None:
        train = pd.concat([train, names], ignore_index=True)

    write_corpus(os.path.join(root_dir, "train.src.txt"), encode_sp_lines(src_spp, train["source"]))
    write_corpus(os.path.join(root_dir, "train.trg.txt"), encode_sp_lines(trg_spp, train["target"]))

    print("Writing validation data set...")
    if len(val) > 0:
        val_src = itertools.chain.from_iterable(map(lambda pair_val: pair_val["source"], val.values()))
        write_corpus(os.path.join(root_dir, "val.src.txt"), encode_sp_lines(src_spp, val_src))
        write_val_corpora(trg_spp, multi_ref_eval, val, root_dir)

    print("Writing test data set...")
    for old_file_path in glob(os.path.join(root_dir, "test.*.txt")):
        os.remove(old_file_path)

    for (src_iso, trg_iso), pair_test in test.items():
        prefix = "test" if len(test) == 1 else f"test.{src_iso}.{trg_iso}"
        write_corpus(os.path.join(root_dir, f"{prefix}.vref.txt"), map(lambda vr: str(vr), pair_test["vref"]))
        write_corpus(os.path.join(root_dir, f"{prefix}.src.txt"), encode_sp_lines(src_spp, pair_test["source"]))

        columns: List[str] = list(filter(lambda c: c.startswith("target"), pair_test.columns))
        for column in columns:
            project = column[len("target_") :]
            trg_suffix = "" if len(columns) == 1 else f".{project}"
            write_corpus(
                os.path.join(root_dir, f"{prefix}.trg{trg_suffix}.txt"), encode_sp_lines(trg_spp, pair_test[column]),
            )
            write_corpus(
                os.path.join(root_dir, f"{prefix}.trg.detok{trg_suffix}.txt"),
                decode_sp_lines(encode_sp_lines(trg_spp, pair_test[column])),
            )


def get_parallel_corpus_length(src_file_path: str, trg_file_path: str) -> int:
    count = 0
    with open(src_file_path, "r", encoding="utf-8") as src_file, open(trg_file_path, "r", encoding="utf-8") as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if len(src_line) > 0 and len(trg_line) > 0:
                count += 1
    return count


def preprocess_standard(
    root_dir: str,
    config: dict,
    src_files: List[DataFile],
    trg_files: List[DataFile],
    write_trg_tag: bool,
    src_spp: Optional[sp.SentencePieceProcessor],
    trg_spp: Optional[sp.SentencePieceProcessor],
) -> None:
    data_config: dict = config["data"]
    test_size: int = data_config["test_size"]
    val_size: int = data_config["val_size"]
    mirror: bool = data_config["mirror"]

#    trg_files_by_project = {df.project: df for df in trg_files}
    test_set_count = 0
    for src_file in src_files:
        if src_file.is_test:
            for trg_file in trg_files:
                if src_file.project == trg_file.project and trg_file.is_test:
                    test_set_count += 1
    #            trg_file = trg_files_by_project.get(src_file.project)
#            if trg_file is not None and trg_file.is_test:
#                test_set_count += 1

    print("Writing data sets...")
    for old_file_path in glob(os.path.join(root_dir, "test.*.txt")):
        os.remove(old_file_path)
    with open(os.path.join(root_dir, "train.src.txt"), "w", encoding="utf-8", newline="\n") as train_src_file, open(
        os.path.join(root_dir, "train.trg.txt"), "w", encoding="utf-8", newline="\n"
    ) as train_trg_file, open(
        os.path.join(root_dir, "val.src.txt"), "w", encoding="utf-8", newline="\n"
    ) as val_src_file, open(
        os.path.join(root_dir, "val.trg.txt"), "w", encoding="utf-8", newline="\n"
    ) as val_trg_file:
        for src_file in src_files:
            test_indices: Optional[Set[int]] = set()
            val_indices: Optional[Set[int]] = set()

            for trg_file in trg_files:
                if (src_file.project != trg_file.project) or (src_file.iso == trg_file.iso):
                    continue

                print(f'Pairing {src_file.path} with {trg_file.path}; mirror: {mirror}')

#            trg_file = trg_files_by_project.get(src_file.project)
#            if trg_file is None:
#                continue

                if len(test_indices) == 0 and len(val_indices) == 0:
                    corpus_len = get_parallel_corpus_length(src_file.path, trg_file.path)

                    if src_file.is_test and trg_file.is_test:
                        if test_size < 0 or test_size >= corpus_len:
                            test_indices = None
                        else:
                            test_indices = set(random.sample(range(corpus_len), test_size))
                    if src_file.is_val and trg_file.is_val and test_indices is not None:
                        population = (
                            range(corpus_len)
                            if len(test_indices) == 0
                            else [i for i in range(corpus_len) if i not in test_indices]
                        )
                        if val_size < 0 or val_size >= len(population):
                            val_indices = None
                        else:
                            val_indices = set(random.sample(population, val_size))

                test_prefix = "test" if test_set_count == 1 else f"test.{src_file.iso}.{trg_file.iso}"
                with open(src_file.path, "r", encoding="utf-8") as input_src_file, open(
                    trg_file.path, "r", encoding="utf-8"
                ) as input_trg_file, open(
                    os.path.join(root_dir, f"{test_prefix}.src.txt"), "a", encoding="utf-8", newline="\n"
                ) as test_src_file, open(
                    os.path.join(root_dir, f"{test_prefix}.trg.detok.txt"), "a", encoding="utf-8", newline="\n"
                ) as test_trg_file:
                    index = 0
                    for src_line, trg_line in zip(input_src_file, input_trg_file):
                        src_line = src_line.strip()
                        trg_line = trg_line.strip()
                        if len(src_line) == 0 or len(trg_line) == 0:
                            continue

                        src_sentence = src_line
                        if write_trg_tag:
                            src_sentence = f"<2{trg_file.iso}> " + src_line
                        trg_sentence = trg_line

                        mirror_src_sentence = trg_line
                        if write_trg_tag:
                            mirror_src_sentence = f"<2{src_file.iso}> " + trg_line
                        mirror_trg_sentence = src_line
                        mirror_src_spp = trg_spp
                        mirror_trg_spp = src_spp

                        if test_indices is None or index in test_indices:
                            test_src_file.write(encode_sp(src_spp, src_sentence) + "\n")
                            test_trg_file.write(decode_sp(encode_sp(trg_spp, trg_sentence)) + "\n")
                        elif val_indices is None or index in val_indices:
                            val_src_file.write(encode_sp(src_spp, src_sentence) + "\n")
                            val_trg_file.write(encode_sp(trg_spp, trg_sentence) + "\n")
                            if mirror:
                                val_src_file.write(encode_sp(mirror_src_spp, mirror_src_sentence) + "\n")
                                val_trg_file.write(encode_sp(mirror_trg_spp, mirror_trg_sentence) + "\n")
                        elif src_file.is_train and trg_file.is_train:
                            train_src_file.write(encode_sp(src_spp, src_sentence) + "\n")
                            train_trg_file.write(encode_sp(trg_spp, trg_sentence) + "\n")
                            if mirror:
                                train_src_file.write(encode_sp(mirror_src_spp, mirror_src_sentence) + "\n")
                                train_trg_file.write(encode_sp(mirror_trg_spp, mirror_trg_sentence) + "\n")

                        index += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesses text corpora into a multilingual data set for OpenNMT-tf"
    )
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--stats", default=False, action="store_true", help="Output corpus statistics")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    root_dir = get_mt_root_dir(exp_name)
    config = load_config(exp_name)
    data_config: dict = config["data"]
    is_scripture: bool = data_config["scripture"]

    set_seed(data_config["seed"])

    src_langs = parse_langs(data_config["src_langs"])
    trg_langs = parse_langs(data_config["trg_langs"])
    src_files = get_data_files(src_langs)
    trg_files = get_data_files(trg_langs)
    src_names_files: List[DataFile] = []
    trg_names_files: List[DataFile] = []
    if is_scripture:
        src_names_files, trg_names_files = get_names_files(src_files, trg_langs)

    mirror: bool = data_config["mirror"]

    parent: Optional[str] = data_config.get("parent")
    parent_config = {}
    parent_data_config = {}
    parent_root_dir = ""
    parent_model_to_use = (
        CheckpointType.BEST
        if data_config["parent_use_best"]
        else CheckpointType.AVERAGE
        if data_config["parent_use_average"]
        else CheckpointType.LAST
    )
    has_parent = False
    parent_use_vocab = False
    if parent is not None:
        parent_config = load_config(parent)
        parent_data_config = parent_config["data"]
        parent_params_config = parent_config["params"]
        parent_use_vocab = data_config["parent_use_vocab"]
        freeze_layers: Optional[List[str]] = parent_params_config.get("freeze_layers")
        # do not freeze any word embeddings layer, because we will update them when we create the parent model
        if freeze_layers is not None:
            parent_params_config["freeze_layers"] = list()
        parent_root_dir = get_mt_root_dir(parent)
        has_parent = True

    write_trg_tag = (
        len(trg_langs) > 1
        or len(parent_data_config.get("trg_langs", [])) > 1
        or mirror
        or parent_data_config.get("mirror", False)
    )
    tag_langs: Optional[Set[str]] = None
    if write_trg_tag:
        tag_langs = trg_langs.keys() | src_langs.keys() if mirror else set(trg_langs.keys())

    src_spp: Optional[sp.SentencePieceProcessor] = None
    trg_spp: Optional[sp.SentencePieceProcessor] = None
    if data_config["tokenize"]:
        src_file_paths: Set[str] = set(map(lambda df: df.path, itertools.chain(src_files, src_names_files)))
        trg_file_paths: Set[str] = set(map(lambda df: df.path, itertools.chain(trg_files, trg_names_files)))
        if data_config["share_vocab"]:
            print("Building shared vocabulary...")
            vocab_size: Optional[int] = data_config.get("vocab_size")
            if vocab_size is None:
                vocab_size = data_config.get("src_vocab_size")
                if vocab_size is None:
                    vocab_size = data_config["trg_vocab_size"]
                elif data_config.get("trg_vocab_size", vocab_size) != vocab_size:
                    raise RuntimeError(
                        "The source and target vocab sizes cannot be different when creating a shared vocab."
                    )

            casing: Optional[str] = data_config.get("casing")
            if casing is None:
                casing = data_config.get("src_casing")
                if casing is None:
                    casing = data_config["trg_casing"]
                elif data_config.get("trg_casing", casing) != casing:
                    raise RuntimeError("The source and target casing cannot be different when creating a shared vocab.")

            model_prefix = os.path.join(root_dir, "sp")
            vocab_path = os.path.join(root_dir, "onmt.vocab")
            share_vocab_file_paths: Set[str] = src_file_paths | trg_file_paths
            character_coverage = data_config.get("character_coverage", 1.0)
            build_vocab(
                share_vocab_file_paths, vocab_size, casing, character_coverage, model_prefix, vocab_path, tag_langs
            )

            if has_parent:
                update_vocab(parent_config, root_dir, vocab_path, vocab_path, parent_model_to_use)

            src_spp = sp.SentencePieceProcessor()
            src_spp.Load(f"{model_prefix}.model")

            trg_spp = src_spp
        else:
            src_vocab_file_paths: Set[str] = set(src_file_paths)
            if mirror:
                src_vocab_file_paths.update(trg_file_paths)
            create_unshared_vocab(
                root_dir,
                data_config,
                parent_root_dir,
                parent_data_config,
                parent_use_vocab,
                src_langs,
                src_vocab_file_paths,
                "source",
                tag_langs=tag_langs,
            )

            trg_vocab_file_paths: Set[str] = set(trg_file_paths)
            if mirror:
                trg_vocab_file_paths.update(src_file_paths)
            create_unshared_vocab(
                root_dir,
                data_config,
                parent_root_dir,
                parent_data_config,
                parent_use_vocab,
                trg_langs,
                trg_vocab_file_paths,
                "target",
            )

            if has_parent:
                update_vocab(
                    parent_config,
                    root_dir,
                    os.path.join(root_dir, "src-onmt.vocab"),
                    os.path.join(root_dir, "trg-onmt.vocab"),
                    parent_model_to_use,
                )

            src_spp = sp.SentencePieceProcessor()
            src_spp.Load(os.path.join(root_dir, "src-sp.model"))

            trg_spp = sp.SentencePieceProcessor()
            trg_spp.Load(os.path.join(root_dir, "trg-sp.model"))

    if data_config["scripture"]:
        preprocess_scripture(
            root_dir,
            config,
            args,
            src_files,
            trg_files,
            src_names_files,
            trg_names_files,
            write_trg_tag,
            src_spp,
            trg_spp,
        )
    else:
        preprocess_standard(root_dir, config, src_files, trg_files, write_trg_tag, src_spp, trg_spp)

    print("Preprocessing completed")


if __name__ == "__main__":
    main()
