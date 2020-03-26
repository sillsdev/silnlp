import argparse
import os
from glob import glob
from itertools import chain
from typing import Iterable, Iterator, List, Set

import sentencepiece as sp
from opennmt import constants
from opennmt.data import Vocab
from sklearn.model_selection import train_test_split

from nlp.common.environment import paratextPreprocessedDir


def get_parallel_corpus(
    src_file_path: str, src_sentences: List[str], trg_file_path: str, trg_sentences: List[str], write_trg_token: bool
) -> None:
    src_iso = get_iso(src_file_path)
    trg_iso = get_iso(trg_file_path)

    if src_iso == trg_iso:
        return

    with open(src_file_path, "r", encoding="utf-8") as src_file, open(trg_file_path, "r", encoding="utf-8") as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if len(src_line) == 0 or len(trg_line) == 0:
                continue
            if write_trg_token:
                src_line = f"<2{trg_iso}> " + src_line
            src_sentences.append(src_line)
            trg_sentences.append(trg_line)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesses text corpora into a multilingual data set for OpenNMT-tf"
    )
    # Task is used as the name of the directory for the results.
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--src", nargs="+", metavar="lang", help="Source language")
    parser.add_argument("--trg", nargs="+", metavar="lang", help="Target language")
    args = parser.parse_args()

    name = args.task
    src_langs: Set[str] = set(args.src)
    trg_langs: Set[str] = set(args.trg)

    root_dir = os.path.join(paratextPreprocessedDir, "tests", name)
    model_prefix = os.path.join(root_dir, "sp")
    write_trg_token = len(trg_langs) > 1

    os.makedirs(root_dir, exist_ok=True)

    src_file_paths: List[str] = list()
    trg_file_paths: List[str] = list()
    for file_path in glob(os.path.join(paratextPreprocessedDir, "data", "*.txt")):
        iso = get_iso(file_path)
        if iso in src_langs:
            src_file_paths.append(file_path)
        if iso in trg_langs:
            trg_file_paths.append(file_path)

    src_file_paths.sort()
    trg_file_paths.sort()
    joined_file_paths = ",".join(chain(src_file_paths, trg_file_paths))

    sp_train_params = (
        f"--normalization_rule_name=nmt_nfkc_cf --input={joined_file_paths} --model_prefix={model_prefix}"
        " --vocab_size=24000 --character_coverage=1.0 --input_sentence_size=1000000 --shuffle_input_sentence=true"
    )

    if write_trg_token:
        trg_tokens = list(map(lambda l: f"<2{l}>", trg_langs))
        joined_trg_tokens = ",".join(trg_tokens)
        sp_train_params += f" --control_symbols={joined_trg_tokens}"

    sp.SentencePieceTrainer.train(sp_train_params)

    special_tokens = [constants.PADDING_TOKEN, constants.START_OF_SENTENCE_TOKEN, constants.END_OF_SENTENCE_TOKEN]

    vocab = Vocab(special_tokens)
    vocab.load(f"{model_prefix}.vocab", "sentencepiece")
    vocab.pad_to_multiple(8)
    vocab.serialize(os.path.join(root_dir, "onmt.vocab"))

    spp = sp.SentencePieceProcessor()
    spp.load(f"{model_prefix}.model")

    src_sentences: List[str] = list()
    trg_sentences: List[str] = list()
    for src_file_path in src_file_paths:
        for trg_file_path in trg_file_paths:
            get_parallel_corpus(src_file_path, src_sentences, trg_file_path, trg_sentences, write_trg_token)

    train_src_sentences, test_src_sentences, train_trg_sentences, test_trg_sentences = train_test_split(
        src_sentences, trg_sentences, test_size=2000, random_state=111
    )

    train_src_sentences, val_src_sentences, train_trg_sentences, val_trg_sentences = train_test_split(
        train_src_sentences, train_trg_sentences, test_size=2000, random_state=111
    )

    write_corpus(os.path.join(root_dir, "train.src.txt"), tokenize_sentences(spp, train_src_sentences))
    write_corpus(os.path.join(root_dir, "train.trg.txt"), tokenize_sentences(spp, train_trg_sentences))

    write_corpus(os.path.join(root_dir, "test.src.txt"), tokenize_sentences(spp, test_src_sentences))
    write_corpus(os.path.join(root_dir, "test.trg.txt"), tokenize_sentences(spp, test_trg_sentences))
    write_corpus(os.path.join(root_dir, "test.trg.detok.txt"), test_trg_sentences)

    write_corpus(os.path.join(root_dir, "val.src.txt"), tokenize_sentences(spp, val_src_sentences))
    write_corpus(os.path.join(root_dir, "val.trg.txt"), tokenize_sentences(spp, val_trg_sentences))


if __name__ == "__main__":
    main()
