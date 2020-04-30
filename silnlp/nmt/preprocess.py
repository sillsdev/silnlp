import argparse
import logging
import os
import shutil
from glob import glob
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

logging.basicConfig()

import opennmt
import sentencepiece as sp
from sklearn.model_selection import train_test_split

from nlp.common.environment import paratextPreprocessedDir
from nlp.nmt.config import create_runner, get_root_dir, load_config


class TestData:
    def __init__(self):
        self.src_sentences: List[str] = []
        self.trg_sentences: List[str] = []


def build_vocab(
    file_paths: Iterable[str], vocab_size: int, model_prefix: str, vocab_path: str, tag_langs: Set[str] = None
) -> None:
    joined_file_paths = ",".join(file_paths)

    control_symbols = ["<range>"]
    if tag_langs is not None:
        control_symbols.append("<2qaa>")
        control_symbols.extend(map(lambda l: f"<2{l}>", tag_langs))
    joined_control_symbols = ",".join(control_symbols)

    sp_train_params = (
        f"--normalization_rule_name=nmt_nfkc_cf --input={joined_file_paths} --model_prefix={model_prefix}"
        f" --vocab_size={vocab_size} --character_coverage=1.0 --input_sentence_size=1000000"
        f" --shuffle_input_sentence=true --control_symbols={joined_control_symbols}"
    )

    sp.SentencePieceTrainer.train(sp_train_params)

    special_tokens = [opennmt.PADDING_TOKEN, opennmt.START_OF_SENTENCE_TOKEN, opennmt.END_OF_SENTENCE_TOKEN]

    vocab = opennmt.data.Vocab(special_tokens)
    with open(f"{model_prefix}.vocab", "r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            token = line.rstrip("\r\n")
            token, _ = token.split("\t")
            if token in ("<unk>", "<s>", "</s>", "<range>"):  # Ignore special tokens
                continue
            vocab.add(token)
    vocab.pad_to_multiple(8)
    vocab.serialize(vocab_path)


def get_parallel_corpus(src_file_path: str, trg_file_path: str) -> Tuple[List[str], List[str]]:
    train_src: List[str] = []
    train_trg: List[str] = []
    with open(src_file_path, "r", encoding="utf-8") as src_file, open(trg_file_path, "r", encoding="utf-8") as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if len(src_line) == 0 or len(trg_line) == 0:
                continue
            if src_line == "<range>" and trg_line == "<range>":
                continue
            if src_line == "<range>":
                train_trg[-1] = train_trg[-1] + " " + trg_line
            elif trg_line == "<range>":
                train_src[-1] = train_src[-1] + " " + src_line
            else:
                train_src.append(src_line)
                train_trg.append(trg_line)
    return train_src, train_trg


def get_or_create(d: dict, key: Any, value_selector: Callable[[], Any]) -> Any:
    value = d.get(key)
    if value is None:
        value = value_selector()
        d[key] = value
    return value


def insert_trg_tag(trg_iso: str, write_trg_tag: bool, has_parent: bool, sentences: Iterable[str]) -> Iterable[str]:
    if write_trg_tag:
        trg_iso = "qaa" if has_parent else trg_iso
        for sentence in sentences:
            yield f"<2{trg_iso}> " + sentence
    else:
        for sentence in sentences:
            yield sentence


def write_corpus(corpus_path: str, sentences: Iterable[str]) -> None:
    with open(corpus_path, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def tokenize_sentences(spp: sp.SentencePieceProcessor, sentences: List[str]) -> Iterator[str]:
    for sentence in sentences:
        prefix = ""
        if sentence.startswith("<2"):
            index = sentence.index(">")
            prefix = sentence[0 : index + 2]
            sentence = sentence[index + 2 :]
        yield prefix + " ".join(spp.encode_as_pieces(sentence))


def get_iso(file_path: str) -> str:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    parts = file_name.split("-")
    return parts[0]


def copy_parent_vocab(prefix: str, parent_data_config: dict, parent_root_dir: str, root_dir: str) -> None:
    if parent_data_config.get("share_vocab", True):
        shutil.copy2(os.path.join(parent_root_dir, "onmt.vocab"), os.path.join(root_dir, f"{prefix}-onmt.vocab"))
        shutil.copy2(os.path.join(parent_root_dir, "sp.vocab"), os.path.join(root_dir, f"{prefix}-sp.vocab"))
        shutil.copy2(os.path.join(parent_root_dir, "sp.model"), os.path.join(root_dir, f"{prefix}-sp.model"))
    else:
        shutil.copy2(os.path.join(parent_root_dir, f"{prefix}-onmt.vocab"), root_dir)
        shutil.copy2(os.path.join(parent_root_dir, f"{prefix}-sp.vocab"), root_dir)
        shutil.copy2(os.path.join(parent_root_dir, f"{prefix}-sp.model"), root_dir)


def update_vocab(parent_config: dict, root_dir: str, src_vocab_path: str, trg_vocab_path: str) -> None:
    parent_runner = create_runner(parent_config)
    parent_runner.update_vocab(os.path.join(root_dir, "parent"), src_vocab_path, trg_vocab_path)


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

    src_langs: Set[str] = set(data_config.get("src_langs", []))
    trg_langs: Set[str] = set(data_config.get("trg_langs", []))
    src_file_paths: List[str] = []
    trg_file_paths: List[str] = []
    for file_path in glob(os.path.join(paratextPreprocessedDir, "data", "*.txt")):
        iso = get_iso(file_path)
        if iso in src_langs:
            src_file_paths.append(file_path)
        if iso in trg_langs:
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

    write_trg_tag = len(trg_langs) > 1 or len(parent_data_config.get("trg_langs", [])) > 1 or mirror
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
        src_model_prefix = os.path.join(root_dir, "src-sp")
        src_vocab_path = os.path.join(root_dir, "src-onmt.vocab")
        if src_langs == set(parent_data_config.get("src_langs", [])):
            copy_parent_vocab("src", parent_data_config, parent_root_dir, root_dir)
        else:
            print("Building source vocabulary...")
            src_vocab_size: int = data_config.get("src_vocab_size", 8000)
            src_vocab_file_paths: Set[str] = set(src_file_paths)
            if mirror:
                src_vocab_file_paths.update(trg_file_paths)
            build_vocab(src_vocab_file_paths, src_vocab_size, src_model_prefix, src_vocab_path, tag_langs)

        trg_model_prefix = os.path.join(root_dir, "trg-sp")
        trg_vocab_path = os.path.join(root_dir, "trg-onmt.vocab")
        if trg_langs == set(parent_data_config.get("trg_langs", [])):
            copy_parent_vocab("trg", parent_config, parent_root_dir, root_dir)
        else:
            print("Building target vocabulary...")
            trg_vocab_size: int = data_config.get("trg_vocab_size", 8000)
            trg_vocab_file_paths: Set[str] = set(trg_file_paths)
            if mirror:
                trg_vocab_file_paths.update(src_file_paths)
            build_vocab(trg_vocab_file_paths, trg_vocab_size, trg_model_prefix, trg_vocab_path)

        if has_parent:
            update_vocab(parent_config, root_dir, src_vocab_path, trg_vocab_path)

        src_spp = sp.SentencePieceProcessor()
        src_spp.load(f"{src_model_prefix}.model")

        trg_spp = sp.SentencePieceProcessor()
        trg_spp.load(f"{trg_model_prefix}.model")

    print("Collecting data sets...")
    test_size: int = data_config.get("test_size", 250)
    val_size: int = data_config.get("val_size", 250)
    train_src_sentences: List[str] = []
    train_trg_sentences: List[str] = []
    test_sentences: Dict[str, TestData] = {}
    val_src_sentences: List[str] = []
    val_trg_sentences: List[str] = []
    for src_file_path in src_file_paths:
        for trg_file_path in trg_file_paths:
            src_iso = get_iso(src_file_path)
            trg_iso = get_iso(trg_file_path)

            if src_iso == trg_iso:
                continue

            train_src, train_trg = get_parallel_corpus(src_file_path, trg_file_path)

            train_src, test_src, train_trg, test_trg = train_test_split(
                train_src, train_trg, test_size=test_size, random_state=111
            )
            train_src, val_src, train_trg, val_trg = train_test_split(
                train_src, train_trg, test_size=val_size, random_state=111
            )

            train_src_sentences.extend(insert_trg_tag(trg_iso, write_trg_tag, has_parent, train_src))
            train_trg_sentences.extend(train_trg)

            test_data: TestData = get_or_create(test_sentences, src_iso, lambda: TestData())
            test_data.src_sentences.extend(insert_trg_tag(trg_iso, write_trg_tag, has_parent, test_src))
            test_data.trg_sentences.extend(test_trg)

            val_src_sentences.extend(insert_trg_tag(trg_iso, write_trg_tag, has_parent, val_src))
            val_trg_sentences.extend(val_trg)

            if mirror:
                train_src_sentences.extend(insert_trg_tag(src_iso, write_trg_tag, has_parent, train_trg))
                train_trg_sentences.extend(train_src)

                val_src_sentences.extend(insert_trg_tag(src_iso, write_trg_tag, has_parent, val_trg))
                val_trg_sentences.extend(val_src)

    print("Writing train data set...")
    write_corpus(os.path.join(root_dir, "train.src.txt"), tokenize_sentences(src_spp, train_src_sentences))
    write_corpus(os.path.join(root_dir, "train.trg.txt"), tokenize_sentences(trg_spp, train_trg_sentences))

    print("Writing test data set...")
    for src_iso, test_data in test_sentences.items():
        prefix = "test" if len(test_sentences) == 1 else f"test.{src_iso}"
        write_corpus(os.path.join(root_dir, f"{prefix}.src.txt"), tokenize_sentences(src_spp, test_data.src_sentences))
        write_corpus(os.path.join(root_dir, f"{prefix}.trg.detok.txt"), test_data.trg_sentences)

    print("Writing validation data set...")
    write_corpus(os.path.join(root_dir, "val.src.txt"), tokenize_sentences(src_spp, val_src_sentences))
    write_corpus(os.path.join(root_dir, "val.trg.txt"), tokenize_sentences(trg_spp, val_trg_sentences))

    print("Preprocessing completed")


if __name__ == "__main__":
    main()
