import os
import subprocess
import tempfile
from statistics import mean
from typing import Dict, Iterable, Iterator, List, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..alignment.fast_align import FastAlign
from .environment import PT_PREPROCESSED_DIR
from .verse_ref import VerseRef


def write_corpus(corpus_path: str, sentences: Iterable[str], append: bool = False) -> None:
    with open(corpus_path, "a" if append else "w", encoding="utf-8", newline="\n") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def load_corpus(input_file: str) -> Iterator[str]:
    with open(input_file, "r", encoding="utf-8-sig") as in_file:
        for line in in_file:
            line = line.strip()
            yield line


def tokenize_corpus(input_path: str, output_path: str) -> None:
    subprocess.run(
        ["dotnet", "machine", "tokenize", input_path, output_path, "-t", "latin", "-l"], stdout=subprocess.DEVNULL,
    )


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

        tokenize_corpus(src_input_path, src_tok_output_path)
        tokenize_corpus(trg_input_path, trg_tok_output_path)

        fast_align = FastAlign(td)

        sym_align_path = os.path.join(td, "sym-align.txt")
        fast_align.align(src_tok_output_path, trg_tok_output_path, sym_align_path)

        forward_prob_table = fast_align.get_forward_prob_table()
        reverse_prob_table = fast_align.get_reverse_prob_table()

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


def get_scripture_parallel_corpus(vref_file_path: str, src_file_path: str, trg_file_path: str) -> pd.DataFrame:
    vrefs: List[VerseRef] = []
    src_sentences: List[str] = []
    trg_sentences: List[str] = []
    indices: List[int] = []
    with open(vref_file_path, "r", encoding="utf-8") as vref_file, open(
        src_file_path, "r", encoding="utf-8"
    ) as src_file, open(trg_file_path, "r", encoding="utf-8") as trg_file:
        index = 0
        for vref_line, src_line, trg_line in zip(vref_file, src_file, trg_file):
            vref_line = vref_line.strip()
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            vref = VerseRef.from_string(vref_line)
            if src_line == "<range>" and trg_line == "<range>":
                if vref.chapter_num == vrefs[-1].chapter_num:
                    vrefs[-1] = VerseRef.from_range(vrefs[-1].simplify(), vref)
            elif src_line == "<range>":
                if vref.chapter_num == vrefs[-1].chapter_num:
                    vrefs[-1] = VerseRef.from_range(vrefs[-1].simplify(), vref)
                if len(trg_line) > 0:
                    if len(trg_sentences[-1]) > 0:
                        trg_sentences[-1] += " "
                    trg_sentences[-1] += trg_line
            elif trg_line == "<range>":
                if vref.chapter_num == vrefs[-1].chapter_num:
                    vrefs[-1] = VerseRef.from_range(vrefs[-1].simplify(), vref)
                if len(src_line) > 0:
                    if len(src_sentences[-1]) > 0:
                        src_sentences[-1] += " "
                    src_sentences[-1] += src_line
            else:
                vrefs.append(vref)
                src_sentences.append(src_line)
                trg_sentences.append(trg_line)
                indices.append(index)
            index += 1

    # remove empty sentences
    for i in range(len(vrefs) - 1, -1, -1):
        if len(src_sentences[i]) == 0 or len(trg_sentences[i]) == 0:
            vrefs.pop(i)
            src_sentences.pop(i)
            trg_sentences.pop(i)
            indices.pop(i)

    data = {"vref": vrefs, "source": src_sentences, "target": trg_sentences}
    return pd.DataFrame(data, index=indices)


def split_parallel_corpus(
    corpus: pd.DataFrame, split_size: int, split_indices: Set[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split: pd.DataFrame
    if split_indices is None:
        if split_size == 0:
            split = pd.DataFrame(columns=corpus.columns)
        elif split_size >= len(corpus):
            split = corpus
            corpus = pd.DataFrame(columns=corpus.columns)
        else:
            corpus, split = train_test_split(corpus, test_size=split_size)
            corpus = corpus.copy()
            split = split.copy()
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
    return os.path.join(PT_PREPROCESSED_DIR, "data", f"{iso}-{project}.txt")


def include_books(corpus: pd.DataFrame, books: Set[int]) -> pd.DataFrame:
    return corpus[corpus.apply(lambda r: r["vref"].book_num in books, axis=1)].copy()


def exclude_books(corpus: pd.DataFrame, books: Set[int]) -> pd.DataFrame:
    return corpus[corpus.apply(lambda r: r["vref"].book_num not in books, axis=1)].copy()
