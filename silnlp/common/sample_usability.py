import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from machine.scripture import ALL_BOOK_IDS

from ..nmt.config import get_mt_exp_dir
from ..nmt.quality_estimation import CANONICAL_ORDER

LOGGER = logging.getLogger(__package__ + ".sample_usability")


def stratified_sample(
    usability_verses_file: Path, sample_size: int, random_state: int = 42, books: List[str] = None
) -> None:
    df = pd.read_csv(usability_verses_file, sep="\t")

    books_suffix = ""
    if books is not None and len(books) > 0:
        df, books_suffix = process_books_argument(df, books, usability_verses_file)

    sample = get_sample(df, sample_size, random_state)

    sample.to_csv(usability_verses_file.parent / f"usability_sample{books_suffix}.tsv", sep="\t", index=False)
    print(
        {
            "mu_pop": df["Usability"].mean(),
            "mu_samp": sample["Usability"].mean(),
            "sd_pop": df["Usability"].std(ddof=1),
            "sd_samp": sample["Usability"].std(ddof=1),
        }
    )

    plt.hist(df["Usability"], bins=20, alpha=0.5, label="Population")
    plt.hist(sample["Usability"], bins=20, alpha=0.5, label="Sample")
    plt.legend()

    # Save to in-memory buffer to avoid Windows path resolution issues
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    with open(usability_verses_file.parent / f"usability_sample_hist{books_suffix}.png", "wb") as hist_file:
        hist_file.write(buf.getvalue())


def process_books_argument(
    df: pd.DataFrame,
    books: List[str],
    usability_verses_file: Path,
) -> str:
    missing_books = set(books) - set(df["Book"].unique())
    if missing_books:
        raise ValueError(
            f"Requested book(s) not found in {usability_verses_file.name}: "
            f"{', '.join(sorted(missing_books, key=lambda b: CANONICAL_ORDER[b]))}"
        )

    books = sorted(books, key=lambda b: CANONICAL_ORDER[b])
    df = df[df["Book"].isin(books)]
    if len(books) <= 5:
        books_suffix = "_" + "_".join(books)
    else:
        books_suffix = f"_{len(books)}_books"

    return df, books_suffix


def get_sample(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size > len(df):
        LOGGER.warning(
            f"The sample size {sample_size} is greater than the dataset size {len(df)}."
            f"Using the dataset size as the sample size."
        )
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    bin_cap = 10 if len(df) < 1000 else 20
    qbins = min(
        max(sample_size // 10, 4),
        bin_cap,
        df["Usability"].nunique(),
    )
    df["bin"] = pd.qcut(df["Usability"], q=qbins, duplicates="drop")
    bin_counts = df["bin"].value_counts().sort_index()
    desired = bin_counts * sample_size / len(df)
    n_per_bin = desired.astype(int)
    shortfall = sample_size - n_per_bin.sum()
    if shortfall > 0:
        remainders = (desired - n_per_bin).sort_values(ascending=False)
        for bin_label in remainders.index[:shortfall]:
            n_per_bin[bin_label] += 1

    def _sample_bin(group: pd.DataFrame) -> pd.DataFrame:
        n = n_per_bin[group.name]
        if n == 0:
            return group.iloc[0:0]
        return group.sample(n=n, random_state=random_state)

    sample = (
        df.groupby("bin", group_keys=False, observed=True)
        .apply(_sample_bin, include_groups=False)
        .reset_index(drop=True)
    )
    sample = sample[["Book", "Chapter", "Verse", "Usability"]]
    sample["book_order"] = sample["Book"].map(CANONICAL_ORDER)
    sample = sample.sort_values(["book_order", "Chapter", "Verse"]).drop(columns=["book_order"]).reset_index(drop=True)

    return sample


def main():
    parser = argparse.ArgumentParser(description="Generate a stratified sample from a usability verses file.")
    parser.add_argument(
        "usability_verses_file",
        type=Path,
        help="Path to the usability verses tsv file relative to MT/experiments/",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=60,
        help="Number of verses to include in the sample.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "--books",
        nargs="+",
        metavar="book_ids",
        help="List of book IDs to include in the sample (e.g., MAT MRK LUK). If not provided, all books are included.",
    )
    args = parser.parse_args()

    sample_size = args.sample_size
    random_state = args.random_state
    books = args.books

    if sample_size <= 0 or not isinstance(sample_size, int):
        raise ValueError("Sample size must be a positive integer.")

    if random_state <= 0 or not isinstance(random_state, int):
        raise ValueError("Random state must be a positive integer.")

    if books is not None and len(books) > 0:
        invalid_books = set(books) - set(ALL_BOOK_IDS)
        if invalid_books:
            raise ValueError(f"Invalid book ID(s): {', '.join(invalid_books)}")

    usability_verses_file = get_mt_exp_dir(args.usability_verses_file)
    if not usability_verses_file.exists():
        raise FileNotFoundError(f"The usability verses file {usability_verses_file} does not exist.")
    stratified_sample(usability_verses_file, args.sample_size, args.random_state, args.books)


if __name__ == "__main__":
    main()
