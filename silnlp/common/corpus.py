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


def tokenize_parallel_corpus(
    src_input_path: str, trg_input_path: str, src_output_path: str, trg_output_path: str
) -> None:
    subprocess.run(
        [
            "dotnet",
            "translator",
            "extract",
            "-s",
            src_input_path,
            "-t",
            trg_input_path,
            "-st",
            "latin",
            "-tt",
            "latin",
            "-l",
            "-so",
            src_output_path,
            "-to",
            trg_output_path,
        ],
        stdout=subprocess.DEVNULL,
    )


def load_prob_table(table_path: str) -> Dict[Tuple[str, str], float]:
    table: Dict[Tuple[str, str], float] = {}
    with open(table_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            row = line.split("\t")
            table[(row[0], row[1])] = math.exp(float(row[2]))
    return table


def compute_alignment_score(
    forward_prob_table: Dict[Tuple[str, str], float],
    reverse_prob_table: Dict[Tuple[str, str], float],
    src_sentence: str,
    trg_sentence: str,
    alignment: str,
) -> float:
    pairs = alignment.split(" ")
    src_words = src_sentence.split(" ")
    trg_words = trg_sentence.split(" ")
    probs: List[float] = []
    unaligned_trg_indices: Set[int] = set(range(len(trg_words)))
    unaligned_src_indices: Set[int] = set(range(len(src_words)))
    for pair in pairs:
        if pair != "":
            parts = pair.split("-")
            i = int(parts[0])
            j = int(parts[1])
            unaligned_src_indices.discard(i)
            unaligned_trg_indices.discard(j)
            src_word = src_words[i]
            trg_word = trg_words[j]
            forward_prob = forward_prob_table.get((src_word, trg_word), 1e-9)
            reverse_prob = reverse_prob_table.get((trg_word, src_word), 1e-9)
            prob = max(forward_prob, reverse_prob)
            probs.append(prob)

    for j in unaligned_trg_indices:
        probs.append(forward_prob_table.get(("<eps>", trg_words[j]), 1e-9))

    for i in unaligned_src_indices:
        probs.append(reverse_prob_table.get(("<eps>", src_words[i]), 1e-9))

    return mean(probs)


def wsl_path(win_path: str) -> str:
    win_path = os.path.normpath(win_path).replace("\\", "\\\\")
    result = subprocess.run(["wsl", "wslpath", "-a", win_path], capture_output=True, encoding="utf-8")
    return result.stdout.strip()


def execute_fast_align(input_path: str, output_path: str, prob_table_path: str, reverse: bool) -> None:
    fast_align_path = os.path.join(os.getenv("FAST_ALIGN_PATH", "."), "fast_align")
    if not os.path.isfile(fast_align_path):
        raise RuntimeError("fast_align is not installed.")

    args: List[str]
    if platform.system() == "Windows":
        args = ["wsl", wsl_path(fast_align_path), "-i", wsl_path(input_path), "-p", wsl_path(prob_table_path)]
    else:
        args = [fast_align_path, "-i", input_path, "-p", prob_table_path]
    args.extend(["-d", "-o", "-v"])
    if reverse:
        args.append("-r")

    with open(output_path, "w") as output_file:
        subprocess.run(args, stdout=output_file, stderr=subprocess.DEVNULL)


def execute_atools(forward_align_path: str, reverse_align_path: str, output_path: str) -> None:
    atools_path = os.path.join(os.getenv("FAST_ALIGN_PATH", "."), "atools")
    if not os.path.isfile(atools_path):
        raise RuntimeError("atools is not installed.")

    args: List[str]
    if platform.system() == "Windows":
        args = ["wsl", wsl_path(atools_path), "-i", wsl_path(forward_align_path), "-j", wsl_path(reverse_align_path)]
    else:
        args = [atools_path, "-i", forward_align_path, "-j", reverse_align_path]
    args.extend(["-c", "grow-diag-final-and"])

    with open(output_path, "w") as output_file:
        subprocess.run(args, stdout=output_file, stderr=subprocess.DEVNULL)


def add_alignment_scores(corpus: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, "src-input.txt")
        trg_path = os.path.join(td, "trg-input.txt")
        write_corpus(src_path, corpus["source"])
        write_corpus(trg_path, corpus["target"])
        scores = compute_alignment_scores(src_path, trg_path)
        corpus["score"] = scores


def compute_alignment_scores(src_input_path: str, trg_input_path: str) -> List[float]:
    with tempfile.TemporaryDirectory() as td:
        src_tok_output_path = os.path.join(td, "tokenize-src-output.txt")
        trg_tok_output_path = os.path.join(td, "tokenize-trg-output.txt")

        tokenize_parallel_corpus(src_input_path, trg_input_path, src_tok_output_path, trg_tok_output_path)

        align_input_path = os.path.join(td, "align-input.txt")

        with open(src_tok_output_path, "r", encoding="utf-8") as src_tok_output_file, open(
            trg_tok_output_path, "r", encoding="utf-8"
        ) as trg_tok_output_file, open(align_input_path, "w", encoding="utf-8", newline="\n") as align_input_file:
            for src_sentence, trg_sentence in zip(src_tok_output_file, trg_tok_output_file):
                align_input_file.write(f"{src_sentence.strip()} ||| {trg_sentence.strip()}\n")

        forward_align_path = os.path.join(td, "forward-align.txt")
        forward_prob_table_path = os.path.join(td, "forward-prob-table.txt")
        execute_fast_align(align_input_path, forward_align_path, forward_prob_table_path, reverse=False)

        reverse_align_path = os.path.join(td, "reverse-align.txt")
        reverse_prob_table_path = os.path.join(td, "reverse-prob-table.txt")
        execute_fast_align(align_input_path, reverse_align_path, reverse_prob_table_path, reverse=True)

        sym_align_path = os.path.join(td, "sym-align.txt")
        execute_atools(forward_align_path, reverse_align_path, sym_align_path)

        forward_prob_table = load_prob_table(forward_prob_table_path)
        reverse_prob_table = load_prob_table(reverse_prob_table_path)

        scores: List[float] = []
        with open(src_tok_output_path, "r", encoding="utf-8") as src_tok_output_file, open(
            trg_tok_output_path, "r", encoding="utf-8"
        ) as trg_tok_output_file, open(sym_align_path, "r", encoding="utf-8") as sym_align_file:
            for src_sentence, trg_sentence, alignment in zip(src_tok_output_file, trg_tok_output_file, sym_align_file):
                scores.append(
                    compute_alignment_score(
                        forward_prob_table, reverse_prob_table, src_sentence, trg_sentence, alignment
                    )
                )
        return scores


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
    if score_threshold < 1:
        # Filter the corpus entries with alignment scores less than the threshold
        score_threshold = min(corpus["score"].quantile(0.1), score_threshold)
        return corpus[corpus["score"] > score_threshold]
    elif score_threshold < len(corpus):
        # Filter <n> corpus entries with the lowest alignment scores (n = score_threshold)
        return corpus.sort_values("score").iloc[int(score_threshold) :]

    return corpus


def get_corpus_path(iso: str, project: str) -> str:
    return os.path.join(paratextPreprocessedDir, "data", f"{iso}-{project}.txt")
