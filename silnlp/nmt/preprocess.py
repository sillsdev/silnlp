import argparse
import bisect
import itertools
import logging
import os
import random
import shutil
from glob import glob
from statistics import mean
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple

logging.basicConfig()

import opennmt.data
import pandas as pd
import sentencepiece as sp

from nlp.common.corpus import (
    add_alignment_scores,
    filter_parallel_corpus,
    get_parallel_corpus,
    split_parallel_corpus,
    write_corpus,
)
from nlp.common.corpus import get_corpus_path, load_corpus
from nlp.common.utils import set_seed
from nlp.nmt.config import create_runner, get_git_revision_hash, get_root_dir, load_config, parse_langs
from nlp.nmt.utils import decode_sp_lines, encode_sp, encode_sp_lines


def convert_vocab(sp_vocab_path: str, onmt_vocab_path: str, tag_langs: Set[str] = None) -> None:
    special_tokens = [opennmt.PADDING_TOKEN, opennmt.START_OF_SENTENCE_TOKEN, opennmt.END_OF_SENTENCE_TOKEN]
    if tag_langs is not None:
        special_tokens.extend(map(lambda l: f"<2{l}>", tag_langs))

    vocab = opennmt.data.Vocab(special_tokens)
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

    sp_train_params = (
        f"--normalization_rule_name={normalization} --input={joined_file_paths} --model_prefix={model_prefix}"
        f" --vocab_size={vocab_size} --character_coverage=1.0 --input_sentence_size=1000000"
        " --shuffle_input_sentence=true --control_symbols=<range>"
    )

    sp.SentencePieceTrainer.Train(sp_train_params)

    convert_vocab(f"{model_prefix}.vocab", vocab_path, tag_langs)


def insert_trg_tag(trg_iso: str, sentences: pd.DataFrame) -> None:
    sentences.loc[:, "source"] = f"<2{trg_iso}> " + sentences.loc[:, "source"]


def get_iso(file_path: str) -> Tuple[str, str]:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    index = file_name.index("-")
    return file_name[:index], file_name[index + 1 :]


def update_vocab(parent_config: dict, root_dir: str, src_vocab_path: str, trg_vocab_path: str) -> None:
    parent_runner = create_runner(parent_config)
    parent_runner.update_vocab(os.path.join(root_dir, "parent"), src_vocab_path, trg_vocab_path)


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
    langs: Set[str],
    vocab_file_paths: Set[str],
    side: str,
    tag_langs: Set[str] = None,
) -> None:
    prefix = "src" if side == "source" else "trg"
    model_prefix = os.path.join(root_dir, f"{prefix}-sp")
    vocab_path = os.path.join(root_dir, f"{prefix}-onmt.vocab")
    parent_langs, _, _ = parse_langs(parent_data_config.get(f"{prefix}_langs", []))
    if langs == parent_langs:
        parent_sp_prefix_path: str
        parent_vocab_path: str
        if parent_data_config["share_vocab"]:
            parent_sp_prefix_path = os.path.join(parent_root_dir, "sp")
            parent_vocab_path = os.path.join(parent_root_dir, "onmt.vocab")
        else:
            parent_sp_prefix_path = os.path.join(parent_root_dir, f"{prefix}-sp")
            parent_vocab_path = os.path.join(parent_root_dir, f"{prefix}-onmt.vocab")

        parent_spp = sp.SentencePieceProcessor()
        parent_spp.Load(parent_sp_prefix_path + ".model")

        parent_vocab = opennmt.data.Vocab()
        parent_vocab.load(parent_vocab_path)

        child_tokens: Set[str] = set()
        for vocab_file_path in vocab_file_paths:
            for line in encode_sp_lines(parent_spp, load_corpus(vocab_file_path)):
                child_tokens.update(line.split())

        # all tokens in the child corpora are in the parent vocab, so we can just use the parent vocab
        if child_tokens.issubset(parent_vocab.words):
            sp_vocab_path = os.path.join(root_dir, f"{prefix}-sp.vocab")
            onmt_vocab_path = os.path.join(root_dir, f"{prefix}-onmt.vocab")
            shutil.copy2(parent_sp_prefix_path + ".model", os.path.join(root_dir, f"{prefix}-sp.model"))
            shutil.copy2(parent_sp_prefix_path + ".vocab", sp_vocab_path)
            convert_vocab(sp_vocab_path, onmt_vocab_path, tag_langs)
            return

    print(f"Building {side} vocabulary...")
    vocab_size: int = data_config.get(f"{prefix}_vocab_size", data_config.get("vocab_size"))
    casing: str = data_config.get(f"{prefix}_casing", data_config.get("casing"))
    build_vocab(vocab_file_paths, vocab_size, casing, model_prefix, vocab_path, tag_langs)


def is_in_sorted(items: list, value: Any) -> bool:
    index = bisect.bisect_left(items, value)
    return index < len(items) and items[index] == value


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesses text corpora into a multilingual data set for OpenNMT-tf"
    )
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--stats", default=False, action="store_true", help="Output corpus statistics")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    root_dir = get_root_dir(exp_name)
    config = load_config(exp_name)
    data_config: dict = config["data"]
    eval_config: dict = config["eval"]
    multi_ref_eval: bool = eval_config["multi_ref_eval"]

    set_seed(data_config["seed"])

    score_threshold: float = data_config["score_threshold"]

    src_langs, src_train_projects, _ = parse_langs(data_config["src_langs"])
    trg_langs, trg_train_projects, trg_test_projects = parse_langs(data_config["trg_langs"])
    src_file_paths: List[str] = []
    trg_file_paths: List[str] = []
    train_only_trg_file_paths: List[str] = []
    test_only_trg_file_paths: List[str] = []
    for src_iso in src_langs:
        for src_train_project in src_train_projects[src_iso]:
            src_file_paths.append(get_corpus_path(src_iso, src_train_project))

    for trg_iso in trg_langs:
        lang_train_projects = trg_train_projects[trg_iso]
        lang_test_projects = trg_test_projects.get(trg_iso)
        for trg_train_project in lang_train_projects:
            trg_file_path = get_corpus_path(trg_iso, trg_train_project)
            trg_file_paths.append(trg_file_path)
            if lang_test_projects is not None and trg_train_project not in lang_test_projects:
                train_only_trg_file_paths.append(trg_file_path)
        if lang_test_projects is not None:
            for trg_test_project in lang_test_projects.difference(lang_train_projects):
                test_only_trg_file_paths.append(get_corpus_path(trg_iso, trg_test_project))

    src_file_paths.sort()
    trg_file_paths.sort()
    train_only_trg_file_paths.sort()
    test_only_trg_file_paths.sort()

    mirror: bool = data_config["mirror"]
    mixed_src: bool = data_config["mixed_src"] and len(src_file_paths) > 1

    parent: Optional[str] = data_config.get("parent")
    parent_config = {}
    parent_data_config = {}
    parent_root_dir = ""
    has_parent = False
    if parent is not None:
        parent_config = load_config(parent)
        parent_data_config = parent_config["data"]
        parent_root_dir = get_root_dir(parent)
        has_parent = True

    write_trg_tag = (
        len(trg_langs) > 1
        or len(parent_data_config.get("trg_langs", [])) > 1
        or mirror
        or parent_data_config.get("mirror", False)
    )
    tag_langs: Optional[Set[str]] = None
    if write_trg_tag:
        tag_langs = trg_langs.union(src_langs) if mirror else trg_langs

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
        share_vocab_file_paths: Set[str] = set(src_file_paths).union(trg_file_paths)
        build_vocab(share_vocab_file_paths, vocab_size, casing, model_prefix, vocab_path, tag_langs)

        if has_parent:
            update_vocab(parent_config, root_dir, vocab_path, vocab_path)

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
            src_langs,
            src_vocab_file_paths,
            "source",
            tag_langs=tag_langs,
        )

        trg_vocab_file_paths: Set[str] = set(trg_file_paths)
        if mirror:
            trg_vocab_file_paths.update(src_file_paths)
        create_unshared_vocab(
            root_dir, data_config, parent_root_dir, parent_data_config, trg_langs, trg_vocab_file_paths, "target"
        )

        if has_parent:
            update_vocab(
                parent_config,
                root_dir,
                os.path.join(root_dir, "src-onmt.vocab"),
                os.path.join(root_dir, "trg-onmt.vocab"),
            )

        src_spp = sp.SentencePieceProcessor()
        src_spp.Load(os.path.join(root_dir, "src-sp.model"))

        trg_spp = sp.SentencePieceProcessor()
        trg_spp.Load(os.path.join(root_dir, "trg-sp.model"))

    print("Collecting data sets...")
    test_size: int = data_config["test_size"]
    val_size: int = data_config["val_size"]
    disjoint_test: bool = data_config["disjoint_test"]
    disjoint_val: bool = data_config["disjoint_val"]

    test_indices: Optional[Set[int]] = None
    val_indices: Optional[Set[int]] = None

    train: Optional[pd.DataFrame] = None
    val: Dict[Tuple[str, str], pd.DataFrame] = {}
    test: Dict[Tuple[str, str], pd.DataFrame] = {}
    pair_val_indices: Dict[Tuple[str, str], Set[int]] = {}
    pair_test_indices: Dict[Tuple[str, str], Set[int]] = {}

    stats_file: Optional[IO] = None
    try:
        if args.stats:
            stats_file = open(os.path.join(root_dir, "corpus-stats.csv"), "w", encoding="utf-8")
            stats_file.write("src_project,trg_project,count,align_score,filtered_count,filtered_align_score\n")

        for src_file_path in src_file_paths:
            for trg_file_path in trg_file_paths + test_only_trg_file_paths:
                src_iso, src_project = get_iso(src_file_path)
                trg_iso, trg_project = get_iso(trg_file_path)

                if src_iso == trg_iso:
                    continue

                is_train_ref = not is_in_sorted(test_only_trg_file_paths, trg_file_path)
                is_test_ref = not is_in_sorted(train_only_trg_file_paths, trg_file_path)

                cur_train = get_parallel_corpus(src_file_path, trg_file_path)
                corpus_len = len(cur_train)
                if is_train_ref and (stats_file is not None or score_threshold > 0):
                    add_alignment_scores(cur_train)
                    if stats_file is not None:
                        cur_train.to_csv(os.path.join(root_dir, f"{src_project}_{trg_project}.csv"), index=False)

                if is_test_ref:
                    if disjoint_test and test_indices is None:
                        indices: Set[int] = set(cur_train.index)
                        if disjoint_val and val_indices is not None:
                            indices.difference_update(val_indices)
                        test_indices = set(random.sample(indices, min(test_size, len(indices))))

                    cur_train, cur_test = split_parallel_corpus(
                        cur_train, test_size, pair_test_indices.get((src_iso, trg_iso), test_indices)
                    )

                    cur_test.drop("score", axis=1, inplace=True, errors="ignore")
                    add_to_eval_dataset(src_iso, trg_iso, trg_project, write_trg_tag, test, pair_test_indices, cur_test)

                if is_train_ref:
                    alignment_score = mean(cur_train["score"]) if stats_file is not None else 0

                    filtered_count = 0
                    filtered_alignment_score = alignment_score
                    if score_threshold > 0:
                        unfiltered_len = len(cur_train)
                        cur_train = filter_parallel_corpus(cur_train, score_threshold)
                        filtered_count = unfiltered_len - len(cur_train)
                        filtered_alignment_score = mean(cur_train["score"]) if stats_file is not None else 0

                    if stats_file is not None:
                        print(f"{src_project} -> {trg_project} stats")
                        print(f"- count: {corpus_len}")
                        print(f"- alignment: {alignment_score:.4f}")
                        print(f"- filtered count: {filtered_count}")
                        print(f"- alignment (filtered): {filtered_alignment_score:.4f}")
                        stats_file.write(
                            f"{src_project},{trg_project},{corpus_len},{alignment_score:.4f},{filtered_count},"
                            f"{filtered_alignment_score:.4f}"
                        )
                    cur_train.drop("score", axis=1, inplace=True, errors="ignore")

                    if disjoint_val and val_indices is None:
                        indices = set(cur_train.index)
                        if disjoint_test and test_indices is not None:
                            indices.difference_update(test_indices)
                        val_indices = set(random.sample(indices, min(val_size, len(indices))))

                    cur_train, cur_val = split_parallel_corpus(
                        cur_train, val_size, pair_val_indices.get((src_iso, trg_iso), val_indices)
                    )

                    if mirror:
                        mirror_cur_train = cur_train.rename(columns={"source": "target", "target": "source"})
                        mirror_cur_val = cur_val.rename(columns={"source": "target", "target": "source"})

                        add_to_eval_dataset(
                            trg_iso, src_iso, src_project, write_trg_tag, val, pair_val_indices, mirror_cur_val
                        )

                        if write_trg_tag:
                            insert_trg_tag(src_iso, mirror_cur_train)

                        train = add_to_train_dataset(trg_project, src_project, mixed_src, train, mirror_cur_train)

                    add_to_eval_dataset(src_iso, trg_iso, trg_project, write_trg_tag, val, pair_val_indices, cur_val)

                    if write_trg_tag:
                        insert_trg_tag(trg_iso, cur_train)

                    train = add_to_train_dataset(src_project, trg_project, mixed_src, train, cur_train)
    finally:
        if stats_file is not None:
            stats_file.close()

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

    write_corpus(os.path.join(root_dir, "train.src.txt"), encode_sp_lines(src_spp, train["source"]))
    write_corpus(os.path.join(root_dir, "train.trg.txt"), encode_sp_lines(trg_spp, train["target"]))

    print("Writing validation data set...")
    val_src = itertools.chain.from_iterable(map(lambda pair_val: pair_val["source"], val.values()))
    write_corpus(os.path.join(root_dir, "val.src.txt"), encode_sp_lines(src_spp, val_src))
    write_val_corpora(trg_spp, multi_ref_eval, val, root_dir)

    print("Writing test data set...")
    for old_file_path in glob(os.path.join(root_dir, "test.*.txt")):
        os.remove(old_file_path)

    for (src_iso, trg_iso), pair_test in test.items():
        prefix = "test" if len(test) == 1 else f"test.{src_iso}.{trg_iso}"
        write_corpus(os.path.join(root_dir, f"{prefix}.src.txt"), encode_sp_lines(src_spp, pair_test["source"]))

        columns: List[str] = list(filter(lambda c: c.startswith("target"), pair_test.columns))
        for column in columns:
            project = column[len("target_") :]
            trg_suffix = "" if len(columns) == 1 else f".{project}"
            write_corpus(
                os.path.join(root_dir, f"{prefix}.trg.detok{trg_suffix}.txt"),
                decode_sp_lines(encode_sp_lines(trg_spp, pair_test[column])),
            )

    print("Preprocessing completed")


if __name__ == "__main__":
    main()
