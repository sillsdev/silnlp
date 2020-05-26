import math
import os
from statistics import mean
import subprocess
import tempfile
from typing import Dict, Iterable, Iterator, List, Tuple

import sentencepiece as sp


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


def sp_tokenize(spp: sp.SentencePieceProcessor, sentences: Iterable[str]) -> Iterator[str]:
    for sentence in sentences:
        prefix = ""
        if sentence.startswith("<2"):
            index = sentence.index(">")
            prefix = sentence[0 : index + 2]
            sentence = sentence[index + 2 :]
        yield prefix + " ".join(spp.encode_as_pieces(sentence))


def load_prob_table(table_path: str) -> Dict[Tuple[str, str], float]:
    table: Dict[Tuple[str, str], float] = {}
    with open(table_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            row = line.split("\t")
            table[(row[0], row[1])] = math.exp(float(row[2]))
    return table


def compute_prob(
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


def alignment_scores(src_sentences: Iterable[str], trg_sentences: Iterable[str]) -> List[float]:
    src_sentences, trg_sentences = tokenize_parallel_corpus(src_sentences, trg_sentences)
    with tempfile.TemporaryDirectory() as td:
        input_path = os.path.join(td, "align-input.txt")
        prob_table_path = os.path.join(td, "prob-table.txt")

        with open(input_path, "w", encoding="utf-8") as file:
            for src_sentence, trg_sentence in zip(src_sentences, trg_sentences):
                file.write(f"{src_sentence} ||| {trg_sentence}\n")

        fast_align_path = os.getenv("FAST_ALIGN_PATH")
        result = subprocess.run(
            [os.path.join(fast_align_path, "fast_align"), "-i", input_path, "-d", "-o", "-v", "-p", prob_table_path,],
            capture_output=True,
            encoding="utf-8",
        )
        output: str = result.stdout
        prob_table = load_prob_table(prob_table_path)

        probs: List[float] = []
        for src_sentence, trg_sentence, alignment in zip(src_sentences, trg_sentences, output.splitlines()):
            probs.append(compute_prob(prob_table, src_sentence, trg_sentence, alignment))
        return probs
