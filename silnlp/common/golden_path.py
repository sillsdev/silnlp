import argparse
import logging
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from machine.corpora import ParallelTextCorpus, escape_spaces, lowercase, nfc_normalize
from machine.scripture import book_id_to_number, book_number_to_id
from machine.tokenization import LatinWordTokenizer, WhitespaceTokenizer
from machine.translation import SymmetrizationHeuristic
from machine.translation.thot import (
    ThotSymmetrizedWordAlignmentModel,
    ThotWordAlignmentModelType,
    create_thot_symmetrized_word_alignment_model,
)

from .corpus import get_mt_corpus_path, get_scripture_parallel_corpus

LOGGER = logging.getLogger(__package__ + ".golden_path")


def get_books(books: List[str]) -> List[int]:
    if isinstance(books, str):
        books = books.split(",")
    result: List[int] = []
    for book_id in books:
        book_id = book_id.strip().strip("*").upper()
        if book_id == "NT":
            result.extend(range(40, 67))
        elif book_id == "OT":
            result.extend(range(40))
        else:
            book_num = book_id_to_number(book_id)
            if book_num is None:
                raise RuntimeError("A specified book Id is invalid.")
            result.append(book_num)
    return result


def preprocess_verse(tokenizer: LatinWordTokenizer, verse: str) -> str:
    return " ".join(lowercase(nfc_normalize(escape_spaces(tokenizer.tokenize(verse)))))


def preprocess(src_path: Path, trg_paths: List[Path], book_nums: List[int]) -> List[pd.DataFrame]:
    tokenizer = LatinWordTokenizer()
    corpora: List[pd.DataFrame] = []
    books = {book_number_to_id(book_num) for book_num in book_nums}
    for trg_path in trg_paths:
        df = get_scripture_parallel_corpus(src_path, trg_path)
        df = df.rename(columns={"vref": "ref"})
        df["text"] = df["ref"].map(lambda vref: vref.book)
        df = df[df["text"].isin(books)]
        df["source"] = df["source"].map(lambda verse: preprocess_verse(tokenizer, verse))
        df["target"] = df["target"].map(lambda verse: preprocess_verse(tokenizer, verse))
        corpora.append(df)
    return corpora


def compute_alignment_prob(
    corpus_df: pd.DataFrame, prev_book_nums: Set[int], next_book_num: int
) -> Tuple[float, float]:
    if next_book_num in prev_book_nums or not corpus_df["text"].eq(book_number_to_id(next_book_num)).any():
        return np.nan, np.nan

    corpus = ParallelTextCorpus.from_pandas(corpus_df)
    model: ThotSymmetrizedWordAlignmentModel = create_thot_symmetrized_word_alignment_model(
        ThotWordAlignmentModelType.FAST_ALIGN
    )
    model.heuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND
    filter_book_nums = prev_book_nums | {next_book_num}
    trainer = model.create_trainer(
        corpus.filter(lambda row: book_id_to_number(row.text_id) in filter_book_nums).tokenize(WhitespaceTokenizer())
    )
    trainer.train()
    trainer.save()

    scores: List[float] = []
    length = 0
    with corpus.filter(lambda row: book_id_to_number(row.text_id) == next_book_num).tokenize(
        WhitespaceTokenizer()
    ).batch(1024) as batches:
        for batch in batches:
            alignments = model.align_batch(batch)
            for row, alignment in zip(batch, alignments):
                score = model.get_avg_translation_score(row.source_segment, row.target_segment, alignment)
                scores.append(score)
                length += 1
    return np.mean(scores).item(), length


def compute_log_probs(
    paths: List[List[int]], corpus_dfs: List[pd.DataFrame], book_nums: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    log_probs: List[np.ndarray] = []
    lengths: List[np.ndarray] = []
    for path in paths:
        prev_book_nums = {book_nums[book_id] for book_id in path}
        path_scores: List[List[float]] = []
        path_lengths: List[List[float]] = []
        for corpus_df in corpus_dfs:
            corpus_scores: List[float] = []
            corpus_lengths: List[float] = []
            for next_book_num in book_nums:
                score, length = compute_alignment_prob(corpus_df, prev_book_nums, next_book_num)
                corpus_scores.append(score)
                corpus_lengths.append(length)
            path_scores.append(corpus_scores)
            path_lengths.append(corpus_lengths)
        if len(path_scores) > 1:
            mean_path_scores = np.nanmean(path_scores, axis=0)
            mean_path_lengths = np.nanmean(path_lengths, axis=0)
        else:
            mean_path_scores = np.array(path_scores[0])
            mean_path_lengths = np.array(path_lengths[0])
        log_probs.append(np.nan_to_num(np.log(mean_path_scores), nan=-np.inf))
        lengths.append(np.nan_to_num(mean_path_lengths))
    return np.stack(log_probs), np.stack(lengths)


def search_step(
    paths: List[List[int]],
    cum_log_probs: np.ndarray,
    cum_lengths: np.ndarray,
    corpus_dfs: List[pd.DataFrame],
    book_nums: List[int],
    beam_width: int,
    length_penalty: float,
) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    log_probs, lengths = compute_log_probs(paths, corpus_dfs, book_nums)
    total_probs = log_probs + cum_log_probs.reshape([-1, 1])
    total_lengths = lengths + cum_lengths.reshape([-1, 1])
    scores = total_probs.copy()
    if length_penalty != 0:
        scores /= np.power((5.0 + total_lengths) / 6.0, length_penalty)
    scores = scores.reshape([-1])
    total_probs = total_probs.reshape([-1])
    total_lengths = total_lengths.reshape([-1])
    sample_ids = top_k(scores, beam_width)
    cum_log_probs = total_probs.take(sample_ids)
    cum_lengths = total_lengths.take(sample_ids)
    book_ids = sample_ids % len(book_nums)
    path_ids = sample_ids // len(book_nums)
    paths = [paths[path_ids[i]] + [book_ids[i]] for i in range(beam_width)]
    return paths, cum_log_probs, cum_lengths


def top_k(array: np.ndarray, k: int) -> np.ndarray:
    indices = np.argpartition(array, -k)[-k:]
    return indices[np.argsort(array[indices])[::-1]]


def beam_search(
    corpus_dfs: List[pd.DataFrame],
    beam_width: int,
    book_nums: List[int],
    start_book_nums: List[int],
    length_penalty: float,
) -> Tuple[List[List[int]], List[float]]:
    paths: List[List[int]] = [[]]
    cum_log_probs = np.zeros(1)
    cum_lengths = np.zeros(1)
    for step in range(len(book_nums)):
        if step < len(start_book_nums):
            cur_beam_width = 1
            cur_book_nums = [start_book_nums[step]]
        else:
            cur_beam_width = beam_width
            cur_book_nums = book_nums
        paths, cum_log_probs, cum_lengths = search_step(
            paths, cum_log_probs, cum_lengths, corpus_dfs, cur_book_nums, cur_beam_width, length_penalty
        )

        LOGGER.info(f"Step {step + 1}")
        for path, score in zip(paths, cum_log_probs.tolist()):
            LOGGER.info(" -> ".join(book_number_to_id(book_nums[book_id]) for book_id in path) + f", {round(score, 8)}")

    return paths, cum_log_probs.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the golden path for a Bible translation")
    parser.add_argument("--corpora", nargs="+", metavar="corpus", help="Corpora")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width")
    parser.add_argument("--books", nargs="*", metavar="book", default=["NT"], help="Books")
    parser.add_argument("--start-books", nargs="*", metavar="book", default=[], help="Starting books")
    parser.add_argument("--length-penalty", type=float, default=0, help="Length penalty")
    args = parser.parse_args()

    src_path = get_mt_corpus_path("grc-GRK")
    trg_paths = [get_mt_corpus_path(c) for c in args.corpora]
    book_nums = get_books(args.books)
    start_book_nums = get_books(args.start_books)
    book_nums.extend(book_num for book_num in start_book_nums if book_num not in book_nums)
    beam_width = min(args.beam_width, len(book_nums))

    corpus_dfs = preprocess(src_path, trg_paths, book_nums)
    paths, scores = beam_search(corpus_dfs, beam_width, book_nums, start_book_nums, args.length_penalty)
    LOGGER.info(
        "Best: "
        + " -> ".join(book_number_to_id(book_nums[book_id]) for book_id in paths[0])
        + f", {round(scores[0], 8)}"
    )


if __name__ == "__main__":
    main()
