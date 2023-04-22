import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from machine.corpora import ParallelTextCorpus, escape_spaces, lowercase, nfc_normalize
from machine.scripture import book_id_to_number, book_number_to_id
from machine.tokenization import LatinWordTokenizer, WhitespaceTokenizer
from machine.translation import SymmetrizationHeuristic
from machine.translation.thot import ThotSymmetrizedWordAlignmentModel, create_thot_symmetrized_word_alignment_model
from tqdm import tqdm

LOGGER = logging.getLogger(__package__ + ".golden_path")


def disable_logging() -> None:
    logging.disable(logging.CRITICAL)


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


def preprocess(src_path: Path, trg_paths: List[Path], book_nums: List[int], preprocess_dir: Path) -> List[Path]:
    from .corpus import get_scripture_parallel_corpus

    tokenizer = LatinWordTokenizer()
    corpora: List[Path] = []
    books = {book_number_to_id(book_num) for book_num in book_nums}
    for trg_path in trg_paths:
        df = get_scripture_parallel_corpus(src_path, trg_path)
        df = df.rename(columns={"vref": "ref"})
        df["text"] = df["ref"].map(lambda vref: vref.book)
        df = df[df["text"].isin(books)]
        df["source"] = df["source"].map(lambda verse: preprocess_verse(tokenizer, verse))
        df["target"] = df["target"].map(lambda verse: preprocess_verse(tokenizer, verse))
        corpus_path = preprocess_dir / (trg_path.stem + ".pkl")
        df.to_pickle(corpus_path)
        corpora.append(corpus_path)
    return corpora


def compute_log_prob(args: Tuple[int, int, List[Path], Set[int], int, str]) -> Tuple[int, int, float, float]:
    i, j, corpus_paths, prev_book_nums, next_book_num, aligner = args
    corpus_scores: List[float] = []
    corpus_lengths: List[float] = []
    for corpus_path in corpus_paths:
        corpus_df: pd.DataFrame = pd.read_pickle(corpus_path)

        if not corpus_df["text"].eq(book_number_to_id(next_book_num)).any():
            continue

        corpus = ParallelTextCorpus.from_pandas(corpus_df)
        model: ThotSymmetrizedWordAlignmentModel = create_thot_symmetrized_word_alignment_model(aligner)
        model.heuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND
        filter_book_nums = prev_book_nums | {next_book_num}
        trainer = model.create_trainer(
            corpus.filter(lambda row: book_id_to_number(row.text_id) in filter_book_nums).tokenize(
                WhitespaceTokenizer()
            )
        )
        trainer.train()
        trainer.save()

        verse_scores: List[float] = []
        book_length = 0.0
        with corpus.filter(lambda row: book_id_to_number(row.text_id) == next_book_num).tokenize(
            WhitespaceTokenizer()
        ).batch(1024) as batches:
            for batch in batches:
                alignments = model.align_batch(batch)
                for row, alignment in zip(batch, alignments):
                    verse_score = model.get_avg_translation_score(row.source_segment, row.target_segment, alignment)
                    verse_scores.append(verse_score)
                    book_length += 1
        corpus_scores.append(np.mean(verse_scores).item())
        corpus_lengths.append(book_length)

    if len(corpus_scores) == 0:
        return i, j, -np.inf, 0.0

    return i, j, np.log(np.mean(corpus_scores, axis=0)), np.mean(corpus_lengths, axis=0)


def compute_log_probs(
    paths: List[List[int]],
    corpus_paths: List[Path],
    book_nums: List[int],
    executor: ProcessPoolExecutor,
    aligner: str,
    force_book_num: int,
) -> Tuple[np.ndarray, np.ndarray]:
    work: List[Tuple[int, int, List[Path], Set[int], int, str]] = []
    for i, path in enumerate(paths):
        prev_book_nums = {book_nums[book_id] for book_id in path}
        for j, next_book_num in enumerate(book_nums):
            if (force_book_num == -1 or force_book_num == next_book_num) and next_book_num not in prev_book_nums:
                work.append((i, j, corpus_paths, prev_book_nums, next_book_num, aligner))

    log_probs = np.full([len(paths), len(book_nums)], -np.inf)
    lengths = np.zeros([len(paths), len(book_nums)])
    for i, j, score, length in tqdm(executor.map(compute_log_prob, work, chunksize=1), total=len(work)):
        log_probs[i, j] = score
        lengths[i, j] = length
    return log_probs, lengths


def search_step(
    paths: List[List[int]],
    cum_log_probs: np.ndarray,
    cum_lengths: np.ndarray,
    corpus_paths: List[Path],
    book_nums: List[int],
    beam_width: int,
    length_penalty: float,
    executor: ProcessPoolExecutor,
    aligner: str,
    force_book_num: int,
) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    log_probs, lengths = compute_log_probs(paths, corpus_paths, book_nums, executor, aligner, force_book_num)
    total_probs = log_probs + cum_log_probs.reshape([-1, 1])
    total_lengths = lengths + cum_lengths.reshape([-1, 1])
    scores = total_probs.copy()
    if length_penalty != 0:
        scores /= np.power((5.0 + total_lengths) / 6.0, length_penalty)
    scores = scores.reshape([-1])
    total_probs = total_probs.reshape([-1])
    total_lengths = total_lengths.reshape([-1])
    actual_beam_width = beam_width
    if force_book_num != -1:
        actual_beam_width = 1
    elif len(scores) < beam_width:
        actual_beam_width = len(scores)
    sample_ids = top_k(scores, actual_beam_width)
    cum_log_probs = total_probs.take(sample_ids)
    cum_lengths = total_lengths.take(sample_ids)
    book_ids = sample_ids % len(book_nums)
    path_ids = sample_ids // len(book_nums)
    paths = [paths[path_ids[i]] + [book_ids[i]] for i in range(actual_beam_width)]
    return paths, cum_log_probs, cum_lengths


def top_k(array: np.ndarray, k: int) -> np.ndarray:
    indices = np.argpartition(array, -k)[-k:]
    return indices[np.argsort(array[indices])[::-1]]


def beam_search(
    corpus_paths: List[Path],
    beam_width: int,
    book_nums: List[int],
    start_book_nums: List[int],
    length_penalty: float,
    executor: ProcessPoolExecutor,
    aligner: str,
) -> Tuple[List[List[int]], List[float]]:
    paths: List[List[int]] = [[]]
    cum_log_probs = np.zeros(1)
    cum_lengths = np.zeros(1)
    for step in range(len(book_nums)):
        LOGGER.info(f"Step {step + 1}")
        paths, cum_log_probs, cum_lengths = search_step(
            paths,
            cum_log_probs,
            cum_lengths,
            corpus_paths,
            book_nums,
            beam_width,
            length_penalty,
            executor,
            aligner,
            start_book_nums[step] if step < len(start_book_nums) else -1,
        )

        for i, (path, score) in enumerate(zip(paths, cum_log_probs.tolist())):
            LOGGER.info(
                f"Path {i + 1}: "
                + " -> ".join(book_number_to_id(book_nums[book_id]) for book_id in path)
                + f", {round(score, 8)}"
            )

    return paths, cum_log_probs.tolist()


def main() -> None:
    from .corpus import get_mt_corpus_path

    parser = argparse.ArgumentParser(description="Compute the golden path for a Bible translation")
    parser.add_argument("--corpora", nargs="+", metavar="corpus", help="Corpora")
    parser.add_argument("--ref-corpus", type=str, default="grc-GRK", help="Reference corpus")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width")
    parser.add_argument("--books", nargs="*", metavar="book", default=["NT"], help="Books")
    parser.add_argument("--start-books", nargs="*", metavar="book", default=[], help="Starting books")
    parser.add_argument("--length-penalty", type=float, default=0, help="Length penalty")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of worker processes")
    parser.add_argument("--aligner", type=str, default="fast_align", help="Aligner")
    args = parser.parse_args()

    src_path = get_mt_corpus_path(args.ref_corpus)
    trg_paths = [get_mt_corpus_path(c) for c in args.corpora]
    book_nums = get_books(args.books)
    start_book_nums = get_books(args.start_books)
    book_nums.extend(book_num for book_num in start_book_nums if book_num not in book_nums)

    with TemporaryDirectory() as temp_dir, ProcessPoolExecutor(
        max_workers=args.max_workers, initializer=disable_logging
    ) as executor:
        preprocess_dir = Path(temp_dir)
        start = time()
        corpus_paths = preprocess(src_path, trg_paths, book_nums, preprocess_dir)
        paths, scores = beam_search(
            corpus_paths, args.beam_width, book_nums, start_book_nums, args.length_penalty, executor, args.aligner
        )
        end = time()
        LOGGER.info(
            "Golden Path: "
            + " -> ".join(book_number_to_id(book_nums[book_id]) for book_id in paths[0])
            + f", {round(scores[0], 8)}"
        )
        LOGGER.info(f"Elapsed time: {timedelta(seconds=end - start)}")


if __name__ == "__main__":
    main()
