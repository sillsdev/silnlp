import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from nltk.translate.api import Alignment

from ..common.canon import ALL_BOOK_IDS, book_id_to_number, get_books, is_ot_nt
from ..common.environment import SIL_NLP_ENV
from ..common.utils import set_seed
from ..common.verse_ref import VerseRef
from .config import get_all_book_paths, load_config
from .lexicon import Lexicon
from .metrics import (
    compute_alignment_metrics,
    compute_lexicon_metrics,
    load_all_alignments,
    load_all_lexicons,
    load_vrefs,
)
from .utils import get_experiment_dirs, get_experiment_name


def add_alignments(dest: Dict[str, List[Alignment]], src: Dict[str, List[Alignment]]) -> None:
    for aligner_id, src_alignments in src.items():
        dest_alignments = dest.get(aligner_id)
        if dest_alignments is None:
            dest[aligner_id] = src_alignments
        else:
            dest_alignments.extend(src_alignments)


def add_lexicons(dest: Dict[str, Lexicon], src: Dict[str, Lexicon]) -> None:
    for aligner_id, src_lexicon in src.items():
        dest_lexicon = dest.get(aligner_id)
        if dest_lexicon is None:
            dest[aligner_id] = src_lexicon
        else:
            dest_lexicon.add(src_lexicon)


def test(exp_dirs: List[Path], by_book: bool, books: Set[int], test_size: Optional[int], output_dir: Path) -> None:
    vrefs: List[VerseRef] = []
    all_alignments: Dict[str, List[Alignment]] = {}
    all_lexicons: Dict[str, Lexicon] = {}
    for exp_dir in exp_dirs:
        vref_file_path = exp_dir / "refs.txt"
        if not vref_file_path.is_file():
            continue
        vrefs += load_vrefs(vref_file_path)
        add_alignments(all_alignments, load_all_alignments(exp_dir))
        add_lexicons(all_lexicons, load_all_lexicons(exp_dir))

    df = compute_alignment_metrics(vrefs, all_alignments, "ALL", books, test_size)
    df = df.join(compute_lexicon_metrics(all_lexicons))

    if by_book:
        for book_id in ALL_BOOK_IDS:
            book_num = book_id_to_number(book_id)
            if not is_ot_nt(book_num) or (len(books) > 0 and book_num not in books):
                continue

            book_df = compute_alignment_metrics(vrefs, all_alignments, book_id, {book_num})
            df = pd.concat([df, book_df])

    for book in sorted(set(df.index.get_level_values("Book")), key=lambda b: 0 if b == "ALL" else book_id_to_number(b)):
        if by_book:
            print(f"--- {book} ---")
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
                ao_at_1: float = row["AO@1"]
                rbo: float = row["RBO"]
                print("Lexicon")
                print(f"- F-Score@1: {f_score_at_1:.4f}")
                print(f"- Precision@1: {precision_at_1:.4f}")
                print(f"- Recall@1: {recall_at_1:.4f}")
                print(f"- F-Score@3: {f_score_at_3:.4f}")
                print(f"- Precision@3: {precision_at_3:.4f}")
                print(f"- Recall@3: {recall_at_3:.4f}")
                print(f"- MAP: {mean_avg_precision:.4f}")
                print(f"- AO@1: {ao_at_1:.4f}")
                print(f"- RBO: {rbo:.4f}")

    scores_file_name = "scores.csv"
    if test_size is not None:
        scores_file_name = f"scores-{test_size}.csv"
    scores_file_path = output_dir / scores_file_name
    df.to_csv(scores_file_path, float_format="%.4f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests generated alignments against gold standard alignments")
    parser.add_argument("experiments", type=str, help="Experiment pattern")
    parser.add_argument("--combine-pattern", type=str, default="*", help="Combine pattern")
    parser.add_argument("--test-size", type=int, help="Test size")
    parser.add_argument("--books", nargs="*", metavar="book", default=[], help="Books")
    parser.add_argument("--by-book", default=False, action="store_true", help="Score individual books")
    args = parser.parse_args()

    books = get_books(args.books)
    test_size: Optional[int] = args.test_size
    if test_size is not None:
        print(f"Test size: {test_size}")

    combine_pattern: str = args.combine_pattern
    combinations: Dict[Path, List[str]] = {}
    for exp_dir in get_experiment_dirs(args.experiments):
        exp_name = get_experiment_name(exp_dir)
        print(f"=== Computing metrics ({exp_name}) ===")
        config = load_config(exp_dir)
        set_seed(config["seed"])
        if config["by_book"]:
            for book, book_exp_dir in get_all_book_paths(exp_dir):
                if book_exp_dir.is_dir():
                    print(f"--- {book} ---")
                    test([book_exp_dir], False, books, test_size, book_exp_dir)
        else:
            test([exp_dir], args.by_book, books, test_size, exp_dir)
        if (
            args.experiments != exp_name
            and exp_dir.parent != SIL_NLP_ENV.align_experiments_dir
            and exp_dir.match(combine_pattern)
        ):
            combination = combinations.get(exp_dir.parent)
            if combination is None:
                combination = []
                combinations[exp_dir.parent] = combination
            combination.append(exp_dir.name)

    for parent_dir, combination in combinations.items():
        exp_name = get_experiment_name(parent_dir) + "/" + "+".join(combination)
        print(f"=== Computing combined metrics ({exp_name}) ===")
        exp_dirs = [parent_dir / name for name in combination]
        test(exp_dirs, args.by_book, books, None, parent_dir)


if __name__ == "__main__":
    main()
