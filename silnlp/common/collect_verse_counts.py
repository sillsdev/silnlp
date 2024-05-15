import argparse
import glob
import os
from collections import Counter
from pathlib import Path
from s3path import S3Path

import pandas as pd
from tqdm import tqdm

from typing import Union

from ..common.environment import SIL_NLP_ENV

OT_canon = [
    "GEN",
    "EXO",
    "LEV",
    "NUM",
    "DEU",
    "JOS",
    "JDG",
    "RUT",
    "1SA",
    "2SA",
    "1KI",
    "2KI",
    "1CH",
    "2CH",
    "EZR",
    "NEH",
    "EST",
    "JOB",
    "PSA",
    "PRO",
    "ECC",
    "SNG",
    "ISA",
    "JER",
    "LAM",
    "EZK",
    "DAN",
    "HOS",
    "JOL",
    "AMO",
    "OBA",
    "JON",
    "MIC",
    "NAM",
    "HAB",
    "ZEP",
    "HAG",
    "ZEC",
    "MAL",
]
DT_canon = [
    "TOB",
    "JDT",
    "ESG",
    "WIS",
    "SIR",
    "BAR",
    "LJE",
    "S3Y",
    "SUS",
    "BEL",
    "1MA",
    "2MA",
    "3MA",
    "4MA",
    "1ES",
    "2ES",
    "MAN",
    "PS2",
    "ODA",
    "PSS",
    "EZA",
    "JUB",
    "ENO",
]
NT_canon = [
    "MAT",
    "MRK",
    "LUK",
    "JHN",
    "ACT",
    "ROM",
    "1CO",
    "2CO",
    "GAL",
    "EPH",
    "PHP",
    "COL",
    "1TH",
    "2TH",
    "1TI",
    "2TI",
    "TIT",
    "PHM",
    "HEB",
    "JAS",
    "1PE",
    "2PE",
    "1JN",
    "2JN",
    "3JN",
    "JUD",
    "REV",
]

def format_path(path: Union[str, Path]):
    output_path = Path(path)
    if not output_path.parent.exists():
        output_path = S3Path(output_path)
        if output_path.parent.exists():
            output_path = f's3:/{output_path}'
    print(f"Writing to {output_path}")
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect various counts from a corpus of Bible extracts")
    parser.add_argument(
        "--input-folder", default=SIL_NLP_ENV.mt_scripture_dir, help="Folder with corpus of Bible extracts"
    )
    parser.add_argument("--output-folder", help="Folder in which to save results", default="", required=False)
    parser.add_argument("--output-exp", help="Experiment folder in which to save results (e.g., FT-Language/Model)", default="", required=False)
    parser.add_argument(
        "--files",
        help="Comma-delimited list of patterns of extract file names to count (e.g. 'arb-*.txt;de-NT.txt)",
        required=True,
    )
    args = parser.parse_args()

    if len(args.output_folder) > 0:
        output_folder = Path(args.output_folder)
    elif len(args.output_exp) > 0:
        output_folder = SIL_NLP_ENV.mt_experiments_dir / args.output_exp
    else:
        raise ValueError("One of --output-exp or --output-folder must be set")

    verse_counts = []
    complete_book_counts = {}
    complete_book_counts_already_collected = False
    extract_files = set()
    for file in args.files.split(";"):
        file = file.strip()
        extract_files_path = Path(args.input_folder, file)
        extract_files_list = glob.glob(str(extract_files_path))
        extract_files = extract_files.union(set(extract_files_list))
        print(f"Processing files with pattern {file}")
        for extract_file_name in tqdm(extract_files_list):
            with open(SIL_NLP_ENV.assets_dir / "vref.txt", "r", encoding="utf-8") as vref_file, open(
                extract_file_name, "r", encoding="utf-8"
            ) as extract_file:
                book_list = []
                chapter_counts = {}
                cur_book = None
                for vref, verse in zip(vref_file, extract_file):
                    cur_book = vref.split(" ")[0]
                    cur_chapter = int(vref.split(" ")[1].split(":")[0].strip())
                    if cur_book not in complete_book_counts:
                        complete_book_counts[cur_book] = []
                    if not complete_book_counts_already_collected:
                        complete_book_counts[cur_book].append(cur_chapter)
                    if verse == "\n":
                        continue
                    if cur_book not in chapter_counts:
                        chapter_counts[cur_book] = []
                    chapter_counts[cur_book].append(cur_chapter)
                    book_list.append(cur_book)
                chapter_counts = {k: Counter(v) for k, v in chapter_counts.items()}
                if not complete_book_counts_already_collected:
                    complete_book_counts = {k: Counter(v) for k, v in complete_book_counts.items()}
                complete_book_counts_already_collected = True
                verse_counts.append(
                    {
                        "file": os.path.basename(extract_file_name),
                        "per_book_counts": Counter(book_list),
                        "per_chapter_counts": chapter_counts,
                    }
                )

    # Initialize the data frames
    verse_count_df = pd.DataFrame(columns=OT_canon + NT_canon + DT_canon)
    verse_count_df["file"] = [os.path.basename(extract_file_name) for extract_file_name in extract_files]
    verse_count_df = verse_count_df.set_index("file")

    verse_percentage_df = pd.DataFrame(columns=OT_canon + NT_canon + DT_canon)
    verse_percentage_df["file"] = [os.path.basename(extract_file_name) for extract_file_name in extract_files]
    verse_percentage_df = verse_percentage_df.set_index("file")

    partially_complete_books = {}

    # Copy the counts to the data frame
    for totals in verse_counts:
        f = totals["file"]
        counts = totals["per_book_counts"]
        for ele in counts:
            verse_count_df.loc[f][ele] = counts[ele]
            verse_percentage_df.loc[f][ele] = 100 * round(counts[ele] / sum(complete_book_counts[ele].values()), 3)
            if verse_percentage_df.loc[f][ele] < 100 and verse_percentage_df.loc[f][ele] > 0:
                if f not in partially_complete_books:
                    partially_complete_books[f] = []
                partially_complete_books[f].append(ele)

    for filename, books in partially_complete_books.items():
        df = pd.DataFrame(
            columns=[i for i in range(1, max([len(complete_book_counts[book].keys()) for book in books]))]
        )
        chapter_counts = list(filter(lambda x: x["file"] == filename, verse_counts))[0]["per_chapter_counts"]
        df["book"] = books
        df = df.set_index("book")
        for book in books:
            for col in df.columns:
                if int(col) <= len(complete_book_counts[book].keys()):
                    df.loc[book][col] = 100 * round(chapter_counts[book][col] / complete_book_counts[book][col], 3)

        df.to_csv(format_path(output_folder / f"{filename[:-4]}_detailed_percentages.csv"))

    verse_count_df.insert(loc=0, column="Books", value=verse_count_df.astype(bool).sum(axis=1))
    verse_count_df.insert(loc=1, column="Total", value=verse_count_df[OT_canon + NT_canon + DT_canon].sum(axis=1))
    verse_count_df.insert(loc=2, column="OT", value=verse_count_df[OT_canon].sum(axis=1))
    verse_count_df.insert(loc=3, column="NT", value=verse_count_df[NT_canon].sum(axis=1))
    verse_count_df.insert(loc=4, column="DT", value=verse_count_df[DT_canon].sum(axis=1))
    verse_count_df.fillna(0, inplace=True)

    verse_percentage_df.insert(
        loc=0, column="Total", value=verse_percentage_df[OT_canon + NT_canon + DT_canon].mean(axis=1).round(1)
    )
    verse_percentage_df.fillna(0.0, inplace=True)  # Replace with 0's before averaging
    verse_percentage_df.insert(loc=1, column="OT", value=verse_percentage_df[OT_canon].mean(axis=1).round(1))
    verse_percentage_df.insert(loc=2, column="NT", value=verse_percentage_df[NT_canon].mean(axis=1).round(1))
    verse_percentage_df.insert(loc=3, column="DT", value=verse_percentage_df[DT_canon].mean(axis=1).round(1))

    verse_count_df.to_csv(format_path(output_folder / "verse_counts.csv"))
    print(verse_count_df)
    verse_percentage_df.to_csv(format_path(output_folder / "verse_percentages.csv"))
    print(verse_percentage_df)
    if len(args.output_exp) > 0:
        print("Copying to bucket...")
        SIL_NLP_ENV.copy_experiment_to_bucket(args.output_exp)

if __name__ == "__main__":
    main()
