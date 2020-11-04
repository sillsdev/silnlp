import argparse
import os
from typing import Optional

from nlp.alignment.config import load_config
from nlp.alignment.metrics import compute_metrics
from nlp.common.canon import get_books
from nlp.common.utils import get_align_root_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests generated alignments against gold standard alignments")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--test-size", type=int, help="Test size")
    parser.add_argument("--books", nargs="*", metavar="book", default=[], help="Books")
    args = parser.parse_args()

    books = get_books(args.books)

    root_dir = get_align_root_dir(args.experiment)
    config = load_config(args.experiment)
    set_seed(config["seed"])
    test_size: Optional[int] = args.test_size

    if test_size is not None:
        print(f"Test size: {test_size}")
    print("Computing metrics...")
    df = compute_metrics(root_dir, books, test_size)

    for name, row in df.iterrows():
        aer: float = row["AER"]
        f_score: float = row["F-Score"]
        precision: float = row["Precision"]
        recall: float = row["Recall"]
        print(name)
        print(f"- AER: {aer:.4f}")
        print(f"- F-Score: {f_score:.4f}")
        print(f"- Precision: {precision:.4f}")
        print(f"- Recall: {recall:.4f}")

    scores_file_name = "scores.csv"
    if test_size is not None:
        scores_file_name = f"scores-{test_size}.csv"
    df.to_csv(os.path.join(root_dir, scores_file_name), float_format="%.4f", index_label="Model")


if __name__ == "__main__":
    main()
