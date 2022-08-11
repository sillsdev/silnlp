import argparse
import logging
import os
import re
import string

import Levenshtein
import pandas as pd
from ..common.corpus import load_corpus
from .utils import decode_sp
from typing import Dict, List
from pathlib import Path
from .config import get_git_revision_hash, get_mt_exp_dir
from collections import Counter

logging.basicConfig()

# Add the em-dash
all_punctuation = string.punctuation + "\u2014"
# Add smart quotes
all_punctuation = all_punctuation + "\u2018" + "\u2019" + "\u201C" + "\u201D"
# Add Danda (0964) and Double Danda (0965) for Devanagari scripts
all_punctuation = all_punctuation + "\u0964" + "\u0965"


def strip_punct(s: str):
    return s.strip(all_punctuation)


def split_punct(s: str):
    return re.findall(all_punctuation, s)


def load_words(word_file: Path, decode: bool = True) -> List[str]:
    unique_words: List[str] = []
    for train_str in load_corpus(word_file):
        if decode:
            line_words = decode_sp(train_str).split(" ")
        else:
            line_words = train_str.split(" ")
        for line_word in line_words:
            stripped_word = strip_punct(line_word.lower())
            if stripped_word is not "" and stripped_word not in unique_words:
                unique_words.append(stripped_word)
    unique_words.sort()
    return unique_words


def load_word_counts(word_file: Path, decode: bool = True) -> Counter:
    all_words: List[str] = []
    for train_str in load_corpus(word_file):
        if decode:
            word_list = decode_sp(train_str).split(" ")
        else:
            word_list = train_str.split(" ")
        for w in word_list:
            stripped_word = strip_punct(w.lower())
            if stripped_word is not "":
                all_words.append(stripped_word)
    all_words.sort()
    word_counts = Counter(all_words)
    return word_counts


def unknown_words(master_words: List[str], these_words: List[str]) -> List[str]:
    unk_words: List[str] = []
    for word in these_words:
        if word not in master_words:
            unk_words.append(word)
    return unk_words


def unknown_word_counts(master_word_counts: Counter, these_word_counts: Counter) -> Counter:
    R1 = master_word_counts & these_word_counts
    R2 = these_word_counts - R1
    return R2


def find_similar_words(master_word_counts: Counter, these_word_counts: Counter, distance: int) -> Dict[str, List[str]]:
    similar_word_list: Dict[str, List[str]] = {}
    for unk_word in sorted(these_word_counts.keys()):
        similar_words: List[str] = []
        for master_word in sorted(master_word_counts.keys()):
            if (
                (Levenshtein.distance(unk_word, master_word) <= distance)
                and unk_word not in master_word
                and master_word not in unk_word
            ):
                similar_words.append(master_word)
        if len(similar_words) > 0:
            similar_word_list[unk_word] = similar_words
    return similar_word_list


def writeWordCounts(writer: pd.ExcelWriter, word_counts: Counter, sheet: str):
    words = pd.DataFrame.from_dict(word_counts, orient="index").reset_index()
    words = words.rename(columns={"index": "word", 0: "count"})
    words.to_excel(writer, index=False, sheet_name=sheet)
    s = writer.sheets[sheet]
    s.autofilter(0, 0, words.shape[0], words.shape[1])


def writeWordList(writer: pd.ExcelWriter, word_list: Dict, sheet: str):
    words = pd.DataFrame.from_dict(word_list, orient="index").reset_index()
    words = words.rename(columns={"index": "word", 0: "similar words"})
    words.to_excel(writer, index=False, sheet_name=sheet)


corpusStats = pd.DataFrame(
    columns=[
        "Corpus",
        "Set",
        "Words (Total)",
        "Words (Unique)",
        "Words (Unknown)",
        "Words (Unknown) %",
        "Words (Unique Unknown)",
        "Words (Unique Unknown) %",
        "Possible Misspellings",
    ]
)


def writeStats(writer: pd.ExcelWriter):
    global corpusStats
    corpusStats.to_excel(writer, index=False, sheet_name="Stats")


def collectStats(
    corpus: str, corpus_set: str, word_counts: Counter, unk_word_counts: Counter, similar_words: Dict[str, List[str]]
):
    global corpusStats

    stats = {
        "Corpus": corpus,
        "Set": corpus_set,
        "Words (Total)": sum(word_counts.values()),
        "Words (Unique)": len(word_counts),
    }
    if unk_word_counts is not None:
        stats["Words (Unknown)"] = sum(unk_word_counts.values())
        stats["Words (Unknown) %"] = sum(unk_word_counts.values()) / sum(word_counts.values()) * 100
        stats["Words (Unique Unknown)"] = len(unk_word_counts)
        stats["Words (Unique Unknown) %"] = len(unk_word_counts) / len(word_counts) * 100
    if similar_words is not None:
        stats["Possible Misspellings"] = len(similar_words)

    corpusStats = corpusStats.append(stats, ignore_index=True)


def wordChecks(
    folder: str,
    writer: pd.ExcelWriter,
    corpus: str,
    similar: bool = False,
    distance: int = 1,
    details: bool = False,
    detok_val: bool = True,
):
    if corpus is not "src" and corpus is not "trg":
        return

    val_similar_words = {}
    test_similar_words = {}

    # Load the training data
    train_file = os.path.join(folder, f"train.{corpus}.txt")
    train_word_counts = load_word_counts(Path(train_file))

    # Load the validation data
    val_file = os.path.join(folder, f"val.{corpus}.txt")
    if detok_val:
        val_word_counts = load_word_counts(Path(val_file))
    else:
        val_word_counts = load_word_counts(Path(val_file), False)
    unk_val_word_counts = unknown_word_counts(train_word_counts, val_word_counts)

    # Load the test data
    test_file = os.path.join(folder, f"test.{corpus}.txt")
    if os.path.exists(test_file):
        test_word_counts = load_word_counts(Path(test_file))
    elif corpus == "trg":
        test_file = os.path.join(folder, f"test.{corpus}.detok.txt")
        test_word_counts = load_word_counts(Path(test_file), False)
    else:
        print(f"No test data for corpus {corpus}")
        return
    unk_test_word_counts = unknown_word_counts(train_word_counts, test_word_counts)
    unk_test_val_word_counts = unknown_word_counts(train_word_counts + val_word_counts, test_word_counts)

    if similar:
        if len(val_word_counts) > 0:
            val_similar_words = find_similar_words(train_word_counts, unk_val_word_counts, distance)
        if len(test_word_counts) > 0:
            test_similar_words = find_similar_words(train_word_counts, unk_test_word_counts, distance)

    collectStats(corpus, "Train", train_word_counts, None, None)
    collectStats(corpus, "Val (vs Train)", val_word_counts, unk_val_word_counts, val_similar_words)
    collectStats(corpus, "Test (vs Train)", test_word_counts, unk_test_word_counts, test_similar_words)
    collectStats(corpus, "Test (vs Train+Val)", test_word_counts, unk_test_val_word_counts, None)

    if details:
        # Source corpora word counts/lists
        writeWordCounts(writer, train_word_counts + val_word_counts + test_word_counts, f"words-all.{corpus}")
        writeWordCounts(writer, train_word_counts, f"words-train.{corpus}")
        if len(val_word_counts) > 0:
            writeWordCounts(writer, val_word_counts, f"words-val.{corpus}")
            writeWordCounts(writer, unk_val_word_counts, f"unknown-val.{corpus} vs train")
            if similar:
                writeWordList(writer, val_similar_words, f"misspelling-val.{corpus}")
        if len(test_word_counts) > 0:
            writeWordCounts(writer, test_word_counts, f"words-test.{corpus}")
            writeWordCounts(writer, unk_test_word_counts, f"unknown-test.{corpus} vs train")
            writeWordCounts(writer, unk_test_val_word_counts, f"unknown-test.{corpus} vs train+val")
            if similar:
                writeWordList(writer, test_similar_words, f"misspelling-test.{corpus}")


def main() -> None:
    writer: pd.ExcelWriter

    parser = argparse.ArgumentParser(description="Checks the train/validate/test splits created for an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--details", default=False, action="store_true", help="Show detailed word lists")
    parser.add_argument("--similar-words", default=False, action="store_true", help="Find similar words")
    parser.add_argument("--distance", default=1, help="Max. Levenshtein distance for word similarity")
    parser.add_argument("--detok-val", default=True, action="store_false", help="Detokenize the target validation set")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    exp_dir = get_mt_exp_dir(exp_name)
    out_file = os.path.join(exp_dir, "word_counts.xlsx")

    pd.set_option("max_columns", None)
    writer = pd.ExcelWriter(out_file, engine="xlsxwriter")
    corpusStats.to_excel(writer, sheet_name="Stats")  # Create empty sheet so that this data is first in the xlsx
    print("Analyzing source ...")
    wordChecks(Path(exp_dir), writer, "src", args.similar_words, args.distance, args.details)
    print("Analyzing target ...")
    wordChecks(Path(exp_dir), writer, "trg", args.similar_words, args.distance, args.details, args.detok_val)
    writeStats(writer)
    writer.save()

    print("Corpus statistics")
    print(corpusStats)

    print(f"Output in {out_file}")


if __name__ == "__main__":
    main()
