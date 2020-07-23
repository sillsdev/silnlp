import math
import os
import platform
import subprocess
import tempfile
from statistics import mean
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from nlp.common.environment import paratextPreprocessedDir


def write_corpus(corpus_path: str, sentences: Iterable[str]) -> None:
    with open(corpus_path, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def load_corpus(input_file: str) -> List[str]:
    sentences: List[str] = []
    with open(input_file, "r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            sentences.append(line)
    return sentences


def tokenize_parallel_corpus(src_sentences: Iterable[str], trg_sentences: Iterable[str]) -> Tuple[List[str], List[str]]:
    with tempfile.TemporaryDirectory() as td:
        src_input_path = os.path.join(td, "tokenize-src-input-1.txt")
        trg_input_path = os.path.join(td, "tokenize-trg-input-1.txt")
        src_output_path = os.path.join(td, "tokenize-src-output-1.txt")
        trg_output_path = os.path.join(td, "tokenize-trg-output-1.txt")

        write_corpus(src_input_path, src_sentences)
        write_corpus(trg_input_path, trg_sentences)

        subprocess.run(
            [
                "dotnet",
                "translator",
                "extract",
                "-s",
                os.path.join(td, "tokenize-src-input-*.txt"),
                "-t",
                os.path.join(td, "tokenize-trg-input-*.txt"),
                "-st",
                "latin",
                "-tt",
                "latin",
                "-l",
                "-so",
                src_output_path,
                "-to",
                trg_output_path,
            ]
        )

        src_sentences = load_corpus(src_output_path)
        trg_sentences = load_corpus(trg_output_path)
        return src_sentences, trg_sentences


def load_prob_table(table_path: str) -> Dict[Tuple[str, str], float]:
    table: Dict[Tuple[str, str], float] = {}
    with open(table_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            row = line.split("\t")
            table[(row[0], row[1])] = math.exp(float(row[2]))
    return table


def compute_score(
    prob_table: Dict[Tuple[str, str], float], src_sentence: str, trg_sentence: str, alignment: str
) -> float:
    pairs = alignment.split(" ")
    src_words = src_sentence.split(" ")
    trg_words = trg_sentence.split(" ")
    probs: List[float] = [0] * len(trg_words)
    for pair in pairs:
        parts = pair.split("-")
        i = int(parts[0])
        j = int(parts[1])
        prob = prob_table.get((src_words[i], trg_words[j]), 1e-9)
        probs[j] = prob if probs[j] == 0 else (probs[j] + prob) / 2

    for j in range(len(trg_words)):
        if probs[j] == 0:
            probs[j] = prob_table.get(("<eps>", trg_words[j]), 1e-9)

    return mean(probs)


def wsl_path(win_path: str) -> str:
    win_path = os.path.normpath(win_path).replace("\\", "\\\\")
    result = subprocess.run(["wsl", "wslpath", "-a", win_path], capture_output=True, encoding="utf-8")
    return result.stdout.strip()


def add_alignment_scores(corpus: pd.DataFrame) -> None:
    fast_align_path = os.path.join(os.getenv("FAST_ALIGN_PATH", "."), "fast_align")
    if not os.path.isfile(fast_align_path):
        raise RuntimeError("fast_align is not installed.")

    src_sentences, trg_sentences = tokenize_parallel_corpus(corpus["source"], corpus["target"])
    with tempfile.TemporaryDirectory() as td:
        input_path = os.path.join(td, "align-input.txt")
        prob_table_path = os.path.join(td, "prob-table.txt")

        with open(input_path, "w", encoding="utf-8", newline="\n") as file:
            for src_sentence, trg_sentence in zip(src_sentences, trg_sentences):
                file.write(f"{src_sentence} ||| {trg_sentence}\n")

        args: List[str]
        if platform.system() == "Windows":
            args = ["wsl", wsl_path(fast_align_path), "-i", wsl_path(input_path), "-p", wsl_path(prob_table_path)]
        else:
            args = [fast_align_path, "-i", input_path, "-p", prob_table_path]
        args.extend(["-d", "-o", "-v"])

        result = subprocess.run(args, capture_output=True, encoding="utf-8")
        output: str = result.stdout
        prob_table = load_prob_table(prob_table_path)

        scores: List[float] = []
        for src_sentence, trg_sentence, alignment in zip(src_sentences, trg_sentences, output.splitlines()):
            scores.append(compute_score(prob_table, src_sentence, trg_sentence, alignment))
        corpus["score"] = scores


def get_parallel_corpus(src_file_path: str, trg_file_path: str) -> pd.DataFrame:
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

    data = {"source": src_sentences, "target": trg_sentences}
    return pd.DataFrame(data, index=indices)


def split_parallel_corpus(
    corpus: pd.DataFrame, split_size: int, split_indices: Set[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split: pd.DataFrame
    if split_indices is None:
        if split_size > len(corpus):
            split = corpus
            corpus = pd.DataFrame(columns=corpus.columns)
        else:
            corpus, split = train_test_split(corpus, test_size=split_size)
    else:
        split = corpus.filter(split_indices, axis=0)
        corpus.drop(split_indices, inplace=True, errors="ignore")
    return corpus, split


def filter_parallel_corpus(corpus: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    score_threshold = min(corpus["score"].quantile(0.1), score_threshold)
    return corpus[corpus["score"] > score_threshold]


def get_corpus_path(iso: str, project: str) -> str:
    return os.path.join(paratextPreprocessedDir, "data", f"{iso}-{project}.txt")
