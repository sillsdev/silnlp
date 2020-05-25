import argparse
import logging
import os
import random
import shutil
from glob import glob
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

logging.basicConfig()

import opennmt
import pandas as pd
import sentencepiece as sp
from sklearn.model_selection import train_test_split

from nlp.common.environment import paratextPreprocessedDir
from nlp.nmt.config import create_runner, get_root_dir, load_config, parse_langs


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

    sp.SentencePieceTrainer.train(sp_train_params)

    convert_vocab(f"{model_prefix}.vocab", vocab_path, tag_langs)


def get_parallel_corpus(
    src_file_path: str,
    trg_file_path: str,
    val_size: int,
    test_size: int,
    val_indices: Set[int] = None,
    test_indices: Set[int] = None,
    random_seed: int = 111,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src_sentences: List[str] = []
    trg_sentences: List[str] = []
    indices: List[int] = []
    with open(src_file_path, "r", encoding="utf-8") as src_file, open(trg_file_path, "r", encoding="utf-8") as trg_file:
        index = 0
        for src_line, trg_line in zip(src_file, trg_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if len(src_line) > 0 and len(trg_line) > 0 and (src_line != "<range>" or trg_line != "<range>"):
                if src_line == "<range>":
                    trg_sentences[-1] = trg_sentences[-1] + " " + trg_line
                elif trg_line == "<range>":
                    src_sentences[-1] = src_sentences[-1] + " " + src_line
                else:
                    src_sentences.append(src_line)
                    trg_sentences.append(trg_line)
                    indices.append(index)
            index += 1

    train = pd.DataFrame({"source": src_sentences, "target": trg_sentences}, index=indices)

    if test_indices is None:
        train, test = train_test_split(train, test_size=test_size, random_state=random_seed)
    else:
        test = train.filter(test_indices, axis=0)
        train.drop(test_indices, inplace=True, errors="ignore")

    if val_indices is None:
        train, val = train_test_split(train, test_size=val_size, random_state=random_seed)
    else:
        val = train.filter(val_indices, axis=0)
        train.drop(val_indices, inplace=True, errors="ignore")

    return train, val, test


def get_or_create(d: dict, key: Any, value_selector: Callable[[], Any]) -> Any:
    value = d.get(key)
    if value is None:
        value = value_selector()
        d[key] = value
    return value


def insert_trg_tag(trg_iso: str, sentences: pd.DataFrame) -> None:
    sentences["source"] = sentences["source"].apply(lambda s: f"<2{trg_iso}> " + s)


def write_corpus(corpus_path: str, sentences: Iterable[str]) -> None:
    with open(corpus_path, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def tokenize_sentences(spp: sp.SentencePieceProcessor, sentences: Iterable[str]) -> Iterator[str]:
    for sentence in sentences:
        prefix = ""
        if sentence.startswith("<2"):
            index = sentence.index(">")
            prefix = sentence[0 : index + 2]
            sentence = sentence[index + 2 :]
        yield prefix + " ".join(spp.encode_as_pieces(sentence))


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
        if projects is None or project in projects:
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
    parent_langs, _ = parse_langs(parent_data_config.get(f"{prefix}_langs", []))
    if langs == parent_langs:
        copy_parent_vocab(prefix, parent_data_config, parent_root_dir, root_dir, tag_langs)
    else:
        print(f"Building {side} vocabulary...")
        vocab_size: int = data_config.get(f"{prefix}_vocab_size", 8000)
        build_vocab(vocab_file_paths, vocab_size, model_prefix, vocab_path, tag_langs)


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

    src_langs, src_lang_projects = parse_langs(data_config.get("src_langs", []))
    trg_langs, trg_lang_projects = parse_langs(data_config.get("trg_langs", []))
    src_file_paths: List[str] = []
    trg_file_paths: List[str] = []
    for file_path in glob(os.path.join(paratextPreprocessedDir, "data", "*.txt")):
        iso, project = get_iso(file_path)
        if is_corpus_in_langs(src_langs, src_lang_projects, iso, project):
            src_file_paths.append(file_path)
        if is_corpus_in_langs(trg_langs, trg_lang_projects, iso, project):
            trg_file_paths.append(file_path)

    src_file_paths.sort()
    trg_file_paths.sort()

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
        src_spp.load(f"{model_prefix}.model")

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
        src_spp.load(os.path.join(root_dir, "src-sp.model"))

        trg_spp = sp.SentencePieceProcessor()
        trg_spp.load(os.path.join(root_dir, "trg-sp.model"))

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

    train = pd.DataFrame(columns=["src_iso", "source", "target"])
    test = pd.DataFrame(columns=["src_iso", "source", "target"])
    val = pd.DataFrame(columns=["src_iso", "source", "target"])
    for src_file_path in src_file_paths:
        for trg_file_path in trg_file_paths:
            src_iso, _ = get_iso(src_file_path)
            trg_iso, _ = get_iso(trg_file_path)

            if src_iso == trg_iso:
                continue

            cur_train, cur_val, cur_test = get_parallel_corpus(
                src_file_path, trg_file_path, val_size, test_size, val_indices, test_indices, seed
            )

            if mirror:
                mirror_cur_train = cur_train.rename(columns={"source": "target", "target": "source"})
                mirror_cur_train["src_iso"] = trg_iso
                mirror_cur_val = cur_val.rename(columns={"source": "target", "target": "source"})
                mirror_cur_val["src_iso"] = trg_iso

                if write_trg_tag:
                    insert_trg_tag(src_iso, mirror_cur_train)
                    insert_trg_tag(src_iso, mirror_cur_val)

                train = pd.concat([train, mirror_cur_train])
                val = pd.concat([val, mirror_cur_val])

            cur_train["src_iso"] = src_iso
            cur_val["src_iso"] = src_iso
            cur_test["src_iso"] = src_iso

            if write_trg_tag:
                insert_trg_tag(trg_iso, cur_train)
                insert_trg_tag(trg_iso, cur_val)
                insert_trg_tag(trg_iso, cur_test)

            train = pd.concat([train, cur_train])
            val = pd.concat([val, cur_val])
            test = pd.concat([test, cur_test])

    print("Writing train data set...")
    write_corpus(os.path.join(root_dir, "train.src.txt"), tokenize_sentences(src_spp, train["source"]))
    write_corpus(os.path.join(root_dir, "train.trg.txt"), tokenize_sentences(trg_spp, train["target"]))

    print("Writing validation data set...")
    write_corpus(os.path.join(root_dir, "val.src.txt"), tokenize_sentences(src_spp, val["source"]))
    write_corpus(os.path.join(root_dir, "val.trg.txt"), tokenize_sentences(trg_spp, val["target"]))

    print("Writing test data set...")
    grouped = test.groupby("src_iso")
    for src_iso, group in grouped:
        prefix = "test" if len(grouped) == 1 else f"test.{src_iso}"
        write_corpus(os.path.join(root_dir, f"{prefix}.src.txt"), tokenize_sentences(src_spp, group["source"]))
        write_corpus(os.path.join(root_dir, f"{prefix}.trg.detok.txt"), group["target"])

    print("Preprocessing completed")


if __name__ == "__main__":
    main()
