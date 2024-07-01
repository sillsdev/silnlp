import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from machine.scripture import ALL_BOOK_IDS, book_id_to_number, is_nt, is_ot
from tqdm import tqdm

from ..common.environment import SIL_NLP_ENV

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


OT_canon = [book for book in ALL_BOOK_IDS if is_ot(book_id_to_number(book))]
NT_canon = [book for book in ALL_BOOK_IDS if is_nt(book_id_to_number(book))]

INCLCUDED_BOOKS = OT_canon + NT_canon + DT_canon
IGNORED_BOOKS = [book for book in ALL_BOOK_IDS if book not in INCLCUDED_BOOKS]


def get_verse_counts(file_path: Path, vref_path: Path):
    verse_counts = Counter()
    with open(vref_path, "r", encoding="utf-8") as vref_file, open(file_path, "r", encoding="utf-8") as extract_file:
        for vref, verse in zip(vref_file, extract_file):
            cur_book = vref.split(" ")[0]
            if verse.strip() and cur_book in INCLCUDED_BOOKS:
                verse_counts[cur_book] += 1
    return verse_counts


def check_for_lock_file(folder: Path, filename: str, file_type: str):
    """Check for lock files and ask the user to close them then exit."""

    if file_type[0] == ".":
        file_type = file_type[1:]

    if file_type.lower() == "csv":
        lockfile = folder / f".~lock.{filename}.{file_type}#"

        if lockfile.is_file():
            print(f"Found lock file: {lockfile}")
            print(
                f"Please close {filename}.{file_type} in folder {folder} in Libre Office Calc OR delete the lock file and try again."
            )
            sys.exit()

    elif file_type.lower() == "xlsx":
        lockfile = folder / f"~${filename}.{file_type}"

        if lockfile.is_file():
            print(f"Found lock file: {lockfile}")
            print(
                f"Please close {filename}.{file_type} in folder {folder} which is open in Excel OR delete the lock file and try again."
            )
            sys.exit()


def check_for_lock_files(folder):
    check_for_lock_file(folder, "verses", "csv")
    check_for_lock_file(folder, "verses", "xlsx")


def count_verses(input_folder, output_folder, recount=False):

    verses_csv = output_folder / "verses" / "verses.csv"
    output_xlsx = verses_csv.with_suffix(".xlsx")

    if verses_csv.is_file() or output_xlsx.is_file():
        check_for_lock_files(output_folder)

    if recount:
        verses_csv.unlink(missing_ok=True)
        output_xlsx.unlink(missing_ok=True)

    if verses_csv.is_file():
        # Read existing results if verses.csv exists
        existing_df = pd.read_csv(verses_csv, index_col=0)

    else:
        # Create an empty dataframe if there isn't a verses.csv file.
        existing_df = pd.DataFrame(columns=["file", "Books", "Total", "OT", "NT", "DT"] + INCLCUDED_BOOKS)
        existing_df.set_index("file", inplace=True)
        
    # Write out the new verses.csv file as a check that it is possible, before counting all the verses
    existing_df.to_csv(verses_csv)

    # Write out the verses.xlsx file as a check that it is possible, before counting all the verses
    existing_df.to_excel(output_xlsx)

    known_files = set(existing_df.index)

    # Step 2: Get all text files in the folder
    text_files = set(Path(f).name for f in glob.glob(str(input_folder / "*.txt")))

    # Step 3: Identify new files to process
    new_files = text_files - known_files
    if recount:
        print(
            f"Recounting verses in {len(new_files)} files."
        )
    else:
        print(
            f"Found {len(new_files)} new files that are not among the {len(known_files)} files already counted in {verses_csv}"
        )

    if new_files:
        # Step 4: Gather verse counts for new files
        results = []
        for file in tqdm(new_files):
            file_path = input_folder / file
            verse_counts = get_verse_counts(file_path, SIL_NLP_ENV.assets_dir / "vref.txt")
            ot_count = sum(verse_counts[book] for book in OT_canon)
            nt_count = sum(verse_counts[book] for book in NT_canon)
            dt_count = sum(verse_counts[book] for book in DT_canon)
            total_count = ot_count + nt_count + dt_count
            book_count = sum(1 for book in INCLCUDED_BOOKS if verse_counts[book] > 0)
            result = {
                "file": file,
                "Books": book_count,
                "Total": total_count,
                "OT": ot_count,
                "NT": nt_count,
                "DT": dt_count,
            }
            result.update(verse_counts)
            results.append(result)

        # Convert results to DataFrame
        if results:
            new_df = pd.DataFrame(results)
            new_df.set_index("file", inplace=True)
            updated_df = pd.concat([existing_df, new_df], axis=0, sort=False)
        else:
            updated_df = existing_df

        # Ensure all columns for all books are present
        for book in INCLCUDED_BOOKS:
            if book not in updated_df.columns:
                updated_df[book] = 0

        # Step 5: Sort results alphabetically by file names ignoring case.
        sorted_df = updated_df.reindex(sorted(updated_df.index, key=lambda x: x.lower()))

        # Step 6: Write out the new verses.csv file, overwriting the previous one
        sorted_df.to_csv(verses_csv)

        # Step 7: Write out the file also as an Excel .xlsx file

        sorted_df.to_excel(output_xlsx)
        print(f"Wrote {len(sorted_df)} verse counts to {verses_csv} and to {output_xlsx}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count the verses in each file from a corpus of Bibles")
    parser.add_argument("--input", default=SIL_NLP_ENV.mt_scripture_dir, help="Folder containing Bibles as text files.")
    parser.add_argument(
        "--output",
        default = SIL_NLP_ENV.mt_experiments_dir / "verses" ,
        help="Folder in which to save results",
    )
    parser.add_argument("--recount", action="store_true", help="Delete existing count files and recount all verses.")

    args = parser.parse_args()
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    print(f"Output folder is {output_folder}")

    recount = args.recount

    # print("All books IDS:")
    # print(ALL_BOOK_IDS)

    # print(f"\nOT books are {OT_canon}")
    # print(f"NT books are {NT_canon}")
    # print(f"DT books are {DT_canon}")
    # print(f"Included books are {INCLCUDED_BOOKS}")
    # print(f"Ignored books are {IGNORED_BOOKS}")

    count_verses(input_folder, output_folder, recount)


if __name__ == "__main__":
    main()
