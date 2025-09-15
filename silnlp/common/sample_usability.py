import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from machine.scripture import ALL_BOOK_IDS

from ..nmt.config import get_mt_exp_dir

LOGGER = logging.getLogger(__package__ + ".sample_usability")


def stratified_sample(
    usability_verses_file: Path, sample_size: int, random_state: int = 42, books: List[str] = None
) -> None:
    df = pd.read_csv(usability_verses_file, sep="\t")
    if books is not None and len(books) > 0:
        invalid_books = set(books) - set(ALL_BOOK_IDS)
        if invalid_books:
            raise ValueError(f"Invalid book ID(s): {', '.join(invalid_books)}")
        df = df[df["Book"].isin(books)]

    if sample_size <= 0 or not isinstance(sample_size, int):
        raise ValueError("Sample size must be a positive integer.")
    if sample_size > len(df):
        LOGGER.warning(
            f"The sample size {sample_size} is greater than the dataset size {len(df)}."
            f"Using the dataset size as the sample size."
        )
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    bin_cap = 10 if len(df) < 1000 else 20
    qbins = min(max(sample_size // 10, 4), bin_cap, df["Usability"].nunique())
    df["bin"] = pd.qcut(df["Usability"], q=qbins, duplicates="drop")
    frac = sample_size / len(df)
    sample = (
        df.groupby("bin", group_keys=False, observed=True)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )
    sample = sample[["Book", "Chapter", "Verse", "Usability"]]
    sample.to_csv(usability_verses_file.parent / "usability_sample.tsv", sep="\t", index=False)
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
    plt.savefig(usability_verses_file.parent / "usability_sample_hist.png")


def main():
    parser = argparse.ArgumentParser(description="Generate a stratified sample from a usability verses file.")
    parser.add_argument(
        "usability_verses_file",
        type=Path,
        help="Path to the usability verses file (TSV format).",
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

    usability_verses_file = get_mt_exp_dir(args.usability_verses_file)
    if not usability_verses_file.exists():
        raise FileNotFoundError(f"The usability verses file {usability_verses_file} does not exist.")
    stratified_sample(usability_verses_file, args.sample_size, args.random_state, args.books)


if __name__ == "__main__":
    main()
