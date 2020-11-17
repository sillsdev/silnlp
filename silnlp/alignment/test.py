import argparse
import os
from typing import Optional, Set

import pandas as pd

from nlp.alignment.config import get_all_book_paths, load_config
from nlp.alignment.metrics import compute_metrics, load_all_alignments, load_vrefs
from nlp.common.canon import ALL_BOOK_IDS, book_id_to_number, get_books, is_ot_nt
from nlp.common.utils import get_align_root_dir, set_seed


def test(root_dir: str, by_book: bool, books: Set[int], test_size: Optional[int]) -> None:
    vref_file_path = os.path.join(root_dir, "refs.txt")
    vrefs = load_vrefs(vref_file_path)

    all_alignments = load_all_alignments(root_dir)

    df = compute_metrics(vrefs, all_alignments, "ALL", books, test_size)

    if by_book:
        for book_id in ALL_BOOK_IDS:
            book_num = book_id_to_number(book_id)
            if not is_ot_nt(book_num) or (len(books) > 0 and book_num not in books):
                continue

            book_df = compute_metrics(vrefs, all_alignments, book_id, {book_num})
            df = pd.concat([df, book_df])

    for book in set(df.index.get_level_values("Book")):
        if by_book:
            print(f"=== {book} ===")
        for index, row in df.loc[[book]].iterrows():
            aer: float = row["AER"]
            f_score: float = row["F-Score"]
            precision: float = row["Precision"]
            recall: float = row["Recall"]
            print(index[1])
            print(f"- AER: {aer:.4f}")
            print(f"- F-Score: {f_score:.4f}")
            print(f"- Precision: {precision:.4f}")
            print(f"- Recall: {recall:.4f}")

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
