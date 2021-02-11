import argparse
import os
from typing import Optional, Set

import pandas as pd

from ..common.canon import ALL_BOOK_IDS, book_id_to_number, get_books, is_ot_nt
from ..common.utils import get_align_root_dir, set_seed
from .config import get_all_book_paths, load_config
from .metrics import (
    compute_alignment_metrics,
    compute_lexicon_metrics,
    load_all_alignments,
    load_all_lexicons,
    load_vrefs,
)


def test(root_dir: str, by_book: bool, books: Set[int], test_size: Optional[int]) -> None:
    vref_file_path = os.path.join(root_dir, "refs.txt")
    vrefs = load_vrefs(vref_file_path)

    all_alignments = load_all_alignments(root_dir)
    df = compute_alignment_metrics(vrefs, all_alignments, "ALL", books, test_size)

    all_lexicons = load_all_lexicons(root_dir)
    df = df.join(compute_lexicon_metrics(root_dir, all_lexicons))

    if by_book:
        for book_id in ALL_BOOK_IDS:
            book_num = book_id_to_number(book_id)
            if not is_ot_nt(book_num) or (len(books) > 0 and book_num not in books):
                continue

            book_df = compute_alignment_metrics(vrefs, all_alignments, book_id, {book_num})
            df = pd.concat([df, book_df])

    for book in sorted(set(df.index.get_level_values("Book")), key=lambda b: 0 if b == "ALL" else book_id_to_number(b)):
        if by_book:
            print(f"=== {book} ===")
        for index, row in df.loc[[book]].iterrows():
            aer: float = row["AER"]
            f_score: float = row["F-Score"]
            precision: float = row["Precision"]
            recall: float = row["Recall"]
            print(f"--- {index[1]} ---")
            print("Alignments")
            print(f"- AER: {aer:.4f}")
            print(f"- F-Score: {f_score:.4f}")
            print(f"- Precision: {precision:.4f}")
            print(f"- Recall: {recall:.4f}")

            if book == "ALL":
                f_score_at_1: float = row["F-Score@1"]
                precision_at_1: float = row["Precision@1"]
                recall_at_1: float = row["Recall@1"]
                f_score_at_3: float = row["F-Score@3"]
                precision_at_3: float = row["Precision@3"]
                recall_at_3: float = row["Recall@3"]
                mean_avg_precision: float = row["MAP"]
                ndcg: float = row["NDCG"]
                rbo: float = row["RBO"]
                print("Lexicon")
                print(f"- F-Score@1: {f_score_at_1:.4f}")
                print(f"- Precision@1: {precision_at_1:.4f}")
                print(f"- Recall@1: {recall_at_1:.4f}")
                print(f"- F-Score@3: {f_score_at_3:.4f}")
                print(f"- Precision@3: {precision_at_3:.4f}")
                print(f"- Recall@3: {recall_at_3:.4f}")
                print(f"- MAP: {mean_avg_precision:.4f}")
                print(f"- NDCG: {ndcg:.4f}")
                print(f"- RBO: {rbo:.4f}")

    scores_file_name = "scores.csv"
    if test_size is not None:
        scores_file_name = f"scores-{test_size}.csv"
    df.to_csv(os.path.join(root_dir, scores_file_name), float_format="%.4f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests generated alignments against gold standard alignments")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--test-size", type=int, help="Test size")
    parser.add_argument("--books", nargs="*", metavar="book", default=[], help="Books")
    parser.add_argument("--by-book", default=False, action="store_true", help="Score individual books")
    args = parser.parse_args()

    books = get_books(args.books)

    root_dir = get_align_root_dir(args.experiment)
    config = load_config(args.experiment)
    set_seed(config["seed"])
    test_size: Optional[int] = args.test_size

    if test_size is not None:
        print(f"Test size: {test_size}")
    print("Computing metrics...")
    if config["by_book"]:
        for book, book_root_dir in get_all_book_paths(root_dir):
            if os.path.isdir(book_root_dir):
                print(f"=== {book} ===")
                test(book_root_dir, False, books, test_size)
    else:
        test(root_dir, args.by_book, books, test_size)


if __name__ == "__main__":
    main()
