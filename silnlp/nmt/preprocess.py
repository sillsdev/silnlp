import argparse
import bisect
import logging
import os
import random
import shutil
from glob import glob
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

logging.basicConfig()

import numpy as np
import opennmt.data
import pandas as pd
import sentencepiece as sp

from nlp.common.environment import paratextPreprocessedDir
from nlp.nmt.config import create_runner, get_root_dir, load_config, parse_langs
from nlp.nmt.corpus import (
    add_alignment_scores,
    filter_parallel_corpus,
    get_parallel_corpus,
    split_parallel_corpus,
    write_corpus,
)


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
    file_paths: Iterable[str], vocab_size: int, model_prefix: str, vocab_path: str, tag_langs: Set[str] = None
) -> None:
    joined_file_paths = ",".join(file_paths)

    sp_train_params = (
        f"--normalization_rule_name=nmt_nfkc_cf --input={joined_file_paths} --model_prefix={model_prefix}"
        f" --vocab_size={vocab_size} --character_coverage=1.0 --input_sentence_size=1000000"
        " --shuffle_input_sentence=true --control_symbols=<range>"
    )

    sp.SentencePieceTrainer.Train(sp_train_params)

    convert_vocab(f"{model_prefix}.vocab", vocab_path, tag_langs)


def sp_tokenize(spp: sp.SentencePieceProcessor, sentences: Iterable[str]) -> Iterator[str]:
    for sentence in sentences:
        prefix = ""
        if sentence.startswith("<2"):
            index = sentence.index(">")
            prefix = sentence[0 : index + 2]
            sentence = sentence[index + 2 :]
        yield prefix + " ".join(spp.EncodeAsPieces(sentence))


def insert_trg_tag(trg_iso: str, sentences: pd.DataFrame) -> None:
    sentences.loc[:, "source"] = f"<2{trg_iso}> " + sentences.loc[:, "source"]


def get_iso(file_path: str) -> Tuple[str, str]:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    index = file_name.index("-")
    return file_name[:index], file_name[index + 1 :]


def copy_parent_vocab(
    prefix: str, parent_data_config: dict, parent_root_dir: str, root_dir: str, tag_langs: Set[str] = None
) -> None:
    sp_vocab_path = os.path.join(root_dir, f"{prefix}-sp.vocab")
    onmt_vocab_path = os.path.join(root_dir, f"{prefix}-onmt.vocab")
    if parent_data_config.get("share_vocab", True):
        shutil.copy2(os.path.join(parent_root_dir, "sp.vocab"), sp_vocab_path)
        shutil.copy2(os.path.join(parent_root_dir, "sp.model"), os.path.join(root_dir, f"{prefix}-sp.model"))
    else:
        shutil.copy2(os.path.join(parent_root_dir, f"{prefix}-sp.vocab"), root_dir)
        shutil.copy2(os.path.join(parent_root_dir, f"{prefix}-sp.model"), root_dir)

    convert_vocab(sp_vocab_path, onmt_vocab_path, tag_langs)


def update_vocab(parent_config: dict, root_dir: str, src_vocab_path: str, trg_vocab_path: str) -> None:
    parent_runner = create_runner(parent_config)
    parent_runner.update_vocab(os.path.join(root_dir, "parent"), src_vocab_path, trg_vocab_path)


def get_sentence_count(file_path: str) -> int:
    lines = 0
    with open(file_path, "r", encoding="utf-8") as file:
        for _ in file:
            lines += 1
    return lines


def is_corpus_in_langs(langs: Set[str], lang_projects: Dict[str, Set[str]], iso: str, project: str) -> bool:
    if iso in langs:
        projects = lang_projects.get(iso)
        if projects is None:
            # do not implicitly include subset corpora
            return "+" not in project and "-" not in project
        elif project in projects:
            return True
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
        copy_parent_vocab(prefix, parent_data_config, parent_root_dir, root_dir, tag_langs)
    else:
        print(f"Building {side} vocabulary...")
        vocab_size: int = data_config.get(f"{prefix}_vocab_size", 8000)
        build_vocab(vocab_file_paths, vocab_size, model_prefix, vocab_path, tag_langs)


def is_in_sorted(items: list, value: Any) -> bool:
    index = bisect.bisect_left(items, value)
    return index < len(items) and items[index] == value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesses text corpora into a multilingual data set for OpenNMT-tf"
    )
    parser.add_argument("experiment", help="Experiment name")

    args = parser.parse_args()

    exp_name = args.experiment
    root_dir = get_root_dir(exp_name)
    config = load_config(exp_name)
    data_config: dict = config.get("data", {})

    seed: Optional[int] = data_config.get("seed")
    if seed is None:
        seed = 111
    random.seed(seed)
    np.random.seed(seed)

    score_threshold: Optional[float] = data_config.get("score_threshold")

    src_langs, src_train_projects, _ = parse_langs(data_config.get("src_langs", []))
    trg_langs, trg_train_projects, trg_test_projects = parse_langs(data_config.get("trg_langs", []))
    src_file_paths: List[str] = []
    trg_file_paths: List[str] = []
    train_only_trg_file_paths: List[str] = []
    test_only_trg_file_paths: List[str] = []
    for file_path in glob(os.path.join(paratextPreprocessedDir, "data", "*.txt")):
        iso, project = get_iso(file_path)
        if is_corpus_in_langs(src_langs, src_train_projects, iso, project):
            src_file_paths.append(file_path)
        if is_corpus_in_langs(trg_langs, trg_train_projects, iso, project):
            trg_file_paths.append(file_path)
            if iso in trg_test_projects and project not in trg_test_projects[iso]:
                train_only_trg_file_paths.append(file_path)
        elif is_corpus_in_langs(trg_langs, trg_test_projects, iso, project):
            test_only_trg_file_paths.append(file_path)

    src_file_paths.sort()
    trg_file_paths.sort()
    train_only_trg_file_paths.sort()
    test_only_trg_file_paths.sort()

    mirror: bool = data_config.get("mirror", False)

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

    if data_config.get("share_vocab", True):
        print("Building shared vocabulary...")
        vocab_size: int = data_config.get("vocab_size", 24000)
        model_prefix = os.path.join(root_dir, "sp")
        vocab_path = os.path.join(root_dir, "onmt.vocab")
        share_vocab_file_paths: Set[str] = set(src_file_paths).union(trg_file_paths)
        build_vocab(share_vocab_file_paths, vocab_size, model_prefix, vocab_path, tag_langs)

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
    test_size: int = data_config.get("test_size", 250)
    val_size: int = data_config.get("val_size", 250)
    disjoint_test: bool = data_config.get("disjoint_test", False)
    disjoint_val: bool = data_config.get("disjoint_val", False)

    test_indices: Optional[Set[int]] = None
    val_indices: Optional[Set[int]] = None
    if disjoint_test or disjoint_val:
        sentence_count = get_sentence_count(src_file_paths[0])
        sample_size = 0
        if disjoint_test:
            sample_size += test_size
        if disjoint_val:
            sample_size += val_size
        samples = random.sample(range(sentence_count), sample_size)
        if disjoint_test:
            test_indices = set(samples[:test_size])
            samples = samples[test_size:]
        if disjoint_val:
            val_indices = set(samples)

    train: Optional[pd.DataFrame] = None
    val: Optional[pd.DataFrame] = None
    test: Optional[pd.DataFrame] = None
    pair_test_indices: Dict[Tuple[str, str], Set[int]] = {}
    ref_indices: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for src_file_path in src_file_paths:
        for trg_file_path in trg_file_paths + test_only_trg_file_paths:
            src_iso, src_project = get_iso(src_file_path)
            trg_iso, trg_project = get_iso(trg_file_path)

            if src_iso == trg_iso:
                continue

            is_train_ref = not is_in_sorted(test_only_trg_file_paths, trg_file_path)
            is_test_ref = not is_in_sorted(train_only_trg_file_paths, trg_file_path)

            cur_train = get_parallel_corpus(src_file_path, trg_file_path)
            if is_train_ref and score_threshold is not None:
                add_alignment_scores(cur_train)

            if is_test_ref:
                cur_train, cur_test = split_parallel_corpus(
                    cur_train, test_size, pair_test_indices.get((src_iso, trg_iso), test_indices)
                )
                if len(cur_test) > 0:
                    if (src_iso, trg_iso) not in pair_test_indices:
                        pair_test_indices[(src_iso, trg_iso)] = set(cur_test.index)

                    cur_test.drop("score", axis=1, inplace=True, errors="ignore")
                    cur_test.set_index(
                        pd.MultiIndex.from_tuples(
                            map(lambda i: (src_iso, trg_iso, i), cur_test.index), names=["src_iso", "trg_iso", "index"]
                        ),
                        inplace=True,
                    )
                    if write_trg_tag:
                        insert_trg_tag(trg_iso, cur_test)

                    ref_index, train_ref_count = ref_indices.get((src_iso, trg_iso), (-1, 0))
                    ref_index += 1
                    if is_train_ref:
                        train_ref_count += 1
                    ref_indices[(src_iso, trg_iso)] = (ref_index, train_ref_count)
                    cur_test = cur_test.rename(columns={"target": f"target_{ref_index}"})

                    test = cur_test if test is None else test.combine_first(cur_test)

            if is_train_ref:
                if score_threshold is not None:
                    unfiltered_len = len(cur_train)
                    cur_train = filter_parallel_corpus(cur_train, score_threshold)
                    print(f"Filtered {unfiltered_len - len(cur_train)} verses from {src_project} -> {trg_project}.")

                cur_train, cur_val = split_parallel_corpus(cur_train, val_size, val_indices)

                if mirror:
                    mirror_cur_train = cur_train.rename(columns={"source": "target", "target": "source"})
                    mirror_cur_val = cur_val.rename(columns={"source": "target", "target": "source"})

                    if write_trg_tag:
                        insert_trg_tag(src_iso, mirror_cur_train)
                        insert_trg_tag(src_iso, mirror_cur_val)

                    train = pd.concat([train, mirror_cur_train], ignore_index=True)
                    val = pd.concat([val, mirror_cur_val], ignore_index=True)

                if write_trg_tag:
                    insert_trg_tag(trg_iso, cur_train)
                    insert_trg_tag(trg_iso, cur_val)

                train = pd.concat([train, cur_train], ignore_index=True)
                val = pd.concat([val, cur_val], ignore_index=True)

    if train is None or val is None or test is None:
        return

    test.fillna("", inplace=True)
    test.sort_index(inplace=True)
    # shuffle train references
    for src_iso in src_langs:
        for trg_iso in trg_langs:
            _, train_ref_count = ref_indices.get((src_iso, trg_iso), (-1, 0))
            if train_ref_count > 1:
                rows = (src_iso, trg_iso, slice(None))
                cols = list(map(lambda i: f"target_{i}", range(train_ref_count)))
                test.loc[rows, cols] = test.loc[rows, cols].apply(
                    lambda r: r.sample(frac=1), result_type="broadcast", axis=1,
                )

    print("Writing train data set...")
    write_corpus(os.path.join(root_dir, "train.src.txt"), sp_tokenize(src_spp, train["source"]))
    write_corpus(os.path.join(root_dir, "train.trg.txt"), sp_tokenize(trg_spp, train["target"]))

    print("Writing validation data set...")
    write_corpus(os.path.join(root_dir, "val.src.txt"), sp_tokenize(src_spp, val["source"]))
    write_corpus(os.path.join(root_dir, "val.trg.txt"), sp_tokenize(trg_spp, val["target"]))

    print("Writing test data set...")
    for old_file_path in glob(os.path.join(root_dir, "test.*.txt")):
        os.remove(old_file_path)
    grouped = test.groupby(level="src_iso")
    for src_iso, group in grouped:
        prefix = "test" if len(grouped) == 1 else f"test.{src_iso}"
        write_corpus(os.path.join(root_dir, f"{prefix}.src.txt"), sp_tokenize(src_spp, group["source"]))

        ref_index = 0
        while f"target_{ref_index}" in test.columns:
            trg_suffix = f".{ref_index}" if "target_1" in test.columns else ""
            write_corpus(os.path.join(root_dir, f"{prefix}.trg.detok{trg_suffix}.txt"), group[f"target_{ref_index}"])
            ref_index += 1

    print("Preprocessing completed")


if __name__ == "__main__":
    main()
