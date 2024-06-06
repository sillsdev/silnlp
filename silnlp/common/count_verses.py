import argparse
import glob
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..common.environment import SIL_NLP_ENV

OT_canon = [
    "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT", "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", 
    "EZR", "NEH", "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER", "LAM", "EZK", "DAN", "HOS", 
    "JOL", "AMO", "OBA", "JON", "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL"
]

DT_canon = [
    "TOB", "JDT", "ESG", "WIS", "SIR", "BAR", "LJE", "S3Y", "SUS", "BEL", "1MA", "2MA", "3MA", "4MA", 
    "1ES", "2ES", "MAN", "PS2", "ODA", "PSS", "EZA", "JUB", "ENO"
]

NT_canon = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH", "PHP", "COL", "1TH", "2TH", 
    "1TI", "2TI", "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV"
]

ALL_BOOKS = OT_canon + NT_canon
IGNORED_BOOKS = DT_canon

def get_verse_counts(file_path: Path, vref_path: Path):
    verse_counts = Counter()
    with open(vref_path, "r", encoding="utf-8") as vref_file, open(file_path, "r", encoding="utf-8") as extract_file:
        for vref, verse in zip(vref_file, extract_file):
            cur_book = vref.split(" ")[0]
            if verse.strip() and cur_book in ALL_BOOKS:
                verse_counts[cur_book] += 1
    return verse_counts


def check_for_lock_file(folder: Path, filename: str, file_type: str):
    """ Check for lock files and ask the user to close them then exit(). """

    import sys
    if file_type[0] == '.':
        file_type = file_type[1:]

    if file_type.lower() == 'csv':
        lockfile = folder / f".~lock.{filename}.{file_type}#"
        
        if lockfile.is_file():
            print(f"Found lock file: {lockfile}")
            print(f"Please close {filename}.{file_type} in folder {folder} in Libre Office Calc OR delete the lock file and try again.")
            sys.exit()
    
    elif file_type.lower() == 'xlsx':
        lockfile =  folder / f"~${filename}.{file_type}"
        
        if lockfile.is_file():
            print(f"Found lock file: {lockfile}")
            print(f"Please close {filename}.{file_type} in folder {folder} which is open in Excel OR delete the lock file and try again.")
            sys.exit()
        
def main() -> None:
    parser = argparse.ArgumentParser(description="Count the verses in each file from a corpus of Bibles")
    parser.add_argument("--folder", default=SIL_NLP_ENV.mt_scripture_dir, help="Folder containing Bibles as text files.")
    parser.add_argument("--output", default=SIL_NLP_ENV.mt_experiments_dir / "verses" / "verses.csv", help="File in which to save results")
    
    args = parser.parse_args()
    folder = Path(args.folder)
    verses_csv = Path(args.output)
    output_folder = verses_csv.parent

    # Step 1: Read existing results if verses.csv exists
    if verses_csv.is_file():
        check_for_lock_file(output_folder, "verses", "csv")
        check_for_lock_file(output_folder, "verses", "xlsx")

        existing_df = pd.read_csv(verses_csv, index_col=0)

    else:
        existing_df = pd.DataFrame(columns=["file", "Books", "Total", "OT", "NT"] + ALL_BOOKS)
        existing_df.set_index("file", inplace=True)

    known_files = set(existing_df.index)
    
    # Step 2: Get all text files in the folder
    text_files = set(Path(f).name for f in glob.glob(str(folder / "*.txt")))

    # Step 3: Identify new files to process
    new_files = text_files - known_files
    print(f"Found {len(new_files)} which are not among the {len(known_files)} files already counted in {verses_csv}")

    # Step 4: Gather verse counts for new files
    results = []
    for file in tqdm(new_files):
        file_path = folder / file
        verse_counts = get_verse_counts(file_path, SIL_NLP_ENV.assets_dir / "vref.txt")
        ot_count = sum(verse_counts[book] for book in OT_canon)
        nt_count = sum(verse_counts[book] for book in NT_canon)
        total_count = ot_count + nt_count
        book_count = sum(1 for book in ALL_BOOKS if verse_counts[book] > 0)
        result = {
            "file": file,
            "Books": book_count,
            "Total": total_count,
            "OT": ot_count,
            "NT": nt_count,
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
    for book in ALL_BOOKS:
        if book not in updated_df.columns:
            updated_df[book] = 0

    # Step 5: Sort results alphabetically by file names ignoring case.
    sorted_df = updated_df.reindex(sorted(updated_df.index, key=lambda x: x.lower()))

    # Step 6: Write out the new verses.csv file, overwriting the previous one
    sorted_df.to_csv(verses_csv)

    # Step 7: Write out the file also as an Excel .xlsx file
    output_xlsx = verses_csv.with_suffix(".xlsx")
    sorted_df.to_excel(output_xlsx)
    print(f"Wrote {len(sorted_df)} verse counts to {verses_csv} and to {output_xlsx}")

if __name__ == "__main__":
    main()
