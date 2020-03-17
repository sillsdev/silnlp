import os
from glob import glob

import sentencepiece as sp
from opennmt import constants
from opennmt.data import Vocab
from sklearn.model_selection import train_test_split

from nlp.common.environment import paratextPreprocessedDir


def get_corpus(data_dir, write_trg_token, src_lang, trg_lang, lang):
    file_path = os.path.join(data_dir, f"all-{src_lang}-{trg_lang}.{lang}")
    if not os.path.exists(file_path):
        file_path = os.path.join(data_dir, f"all-{trg_lang}-{src_lang}.{lang}")

    sentences = list()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if write_trg_token and lang == src_lang:
                line = f"<2{trg_lang}> " + line
            sentences.append(line)
    return sentences


def write_corpus(corpus_path, sentences):
    with open(corpus_path, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def tokenize_sentences(spp, sentences):
    for sentence in sentences:
        prefix = ""
        if sentence.startswith("<2"):
            index = sentence.index(">")
            prefix = sentence[0 : index + 2]
            sentence = sentence[index + 2 :]
        yield prefix + " ".join(spp.encode_as_pieces(sentence))


def main():
    # name = "n-to-1"
    # src_langs = {"bru", "ctu", "cuk", "ifa", "kek", "mps", "nch", "qxn", "rop", "xon"}
    # trg_langs = {"en"}

    name = "1-to-n"
    src_langs = {"en"}
    trg_langs = {"bru", "ctu", "cuk", "ifa", "kek", "mps", "nch", "qxn", "rop", "xon"}

    root_dir = os.path.join(paratextPreprocessedDir, name)
    model_prefix = os.path.join(root_dir, "sp")
    write_trg_token = len(trg_langs) > 1

    os.makedirs(root_dir, exist_ok=True)

    file_paths = list()
    for file_path in glob(os.path.join(paratextPreprocessedDir, "all-*.*")):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        parts = file_name.split("-")
        lang1 = parts[1]
        lang2 = parts[2]
        if (lang1 in src_langs and lang2 in trg_langs) or (lang1 in trg_langs and lang2 in src_langs):
            file_paths.append(file_path)
    joined_file_paths = ",".join(file_paths)

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

    src_sentences = list()
    trg_sentences = list()
    for src_lang in src_langs:
        for trg_lang in trg_langs:
            if src_lang == trg_lang:
                continue
            src_corpus = get_corpus(paratextPreprocessedDir, write_trg_token, src_lang, trg_lang, src_lang)
            src_sentences.extend(src_corpus)
            trg_corpus = get_corpus(paratextPreprocessedDir, write_trg_token, src_lang, trg_lang, trg_lang)
            trg_sentences.extend(trg_corpus)

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
