import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from machine.scripture import ALL_BOOK_IDS, book_id_to_number, is_nt, is_ot
from tqdm import tqdm

from ..common.environment import SIL_NLP_ENV

LOGGER = logging.getLogger(__package__ + ".collect_verse_counts")

OT_CANON = [book for book in ALL_BOOK_IDS if is_ot(book_id_to_number(book))]
NT_CANON = [book for book in ALL_BOOK_IDS if is_nt(book_id_to_number(book))]
DT_CANON = [
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


def get_complete_verse_counts() -> Dict[str, Counter]:
    complete_counts_path = SIL_NLP_ENV.mt_experiments_dir / "verse_counts" / "complete_counts.csv"
    if complete_counts_path.is_file():
        df = pd.read_csv(complete_counts_path, index_col="book").rename(columns=lambda x: int(x))
        return {book: Counter(dict(counts.dropna())) for book, counts in df.iterrows()}

    verse_counts = defaultdict(list)
    with open(SIL_NLP_ENV.assets_dir / "vref.txt", "r", encoding="utf-8") as vref_file:
        for vref in vref_file:
            cur_book = vref.split(" ")[0]
            cur_chapter = int(vref.split(" ")[1].split(":")[0].strip())
            verse_counts[cur_book].append(cur_chapter)
        verse_counts = {k: Counter(v) for k, v in verse_counts.items()}

    # Write to .csv
    max_chapters = max([len(verse_counts[book].keys()) for book in verse_counts.keys()])
    df = pd.DataFrame(columns=[i for i in range(1, max_chapters + 1)])
    df["book"] = verse_counts.keys()
    df = df.set_index("book")
    for book, counts in verse_counts.items():
        for chapter, count in counts.items():
            df.loc[book][chapter] = count
    df.to_csv(complete_counts_path)

    return verse_counts


def collect_verse_counts(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    file_patterns: str,
    deutero: bool = False,
    recount: bool = False,
) -> None:
    input_path = input_folder if isinstance(input_folder, Path) else Path(input_folder)
    output_path = output_folder if isinstance(output_folder, Path) else Path(output_folder)

    extract_files = set()
    for file_pattern in file_patterns.split(";"):
        file_pattern = file_pattern.strip()
        extract_files.update(input_path.glob(file_pattern))
    project_names = [f.stem for f in extract_files]
    projects_to_process = project_names

    bucket_files = ["verse_counts.csv", "verse_percentages.csv", "complete_counts.csv"]
    SIL_NLP_ENV.copy_experiment_from_bucket("verse_counts", bucket_files)
    SIL_NLP_ENV.copy_experiment_from_bucket(
        "verse_counts/partially_complete_books", [f"{f}.csv" for f in project_names]
    )
    partial_books_path = SIL_NLP_ENV.mt_experiments_dir / "verse_counts" / "partially_complete_books"
    partial_books_path.mkdir(exist_ok=True, parents=True)

    # Initialize the data frames and determine which files need to be processed
    verse_counts_path = SIL_NLP_ENV.mt_experiments_dir / "verse_counts" / "verse_counts.csv"
    if verse_counts_path.is_file():
        verse_counts_df = pd.read_csv(verse_counts_path, index_col="file")
        if recount:
            verse_counts_df = verse_counts_df.drop(index=project_names, errors="ignore")
        projects_to_process = list(set(project_names) - set(verse_counts_df.index))
        verse_counts_df = verse_counts_df.reindex(set(verse_counts_df.index) | set(project_names))
    else:
        verse_counts_df = pd.DataFrame(columns=["Books", "Total", "OT", "NT", "DT"] + OT_CANON + NT_CANON + DT_CANON)
        verse_counts_df["file"] = project_names
        verse_counts_df = verse_counts_df.set_index("file")

    verse_percentages_path = SIL_NLP_ENV.mt_experiments_dir / "verse_counts" / "verse_percentages.csv"
    if verse_percentages_path.is_file():
        verse_percentages_df = pd.read_csv(verse_percentages_path, index_col="file")
        if recount:
            verse_percentages_df = verse_percentages_df.drop(index=project_names, errors="ignore")
        verse_percentages_df = verse_percentages_df.reindex(set(verse_percentages_df.index) | set(project_names))
    else:
        verse_percentages_df = pd.DataFrame(columns=["Total", "OT", "NT", "DT"] + OT_CANON + NT_CANON + DT_CANON)
        verse_percentages_df["file"] = project_names
        verse_percentages_df = verse_percentages_df.set_index("file")

    # Get counts for unprocessed files
    complete_verse_counts = get_complete_verse_counts()
    partially_complete_projects = []
    for extract_file_name in tqdm(extract_files):
        project_name = extract_file_name.stem
        if project_name not in projects_to_process:
            LOGGER.info(f"Found verse counts for {project_name}")
            continue
        LOGGER.info(f"Processing {project_name}")

        verse_counts = defaultdict(list)
        with open(SIL_NLP_ENV.assets_dir / "vref.txt", "r", encoding="utf-8") as vref_file, extract_file_name.open(
            "r", encoding="utf-8"
        ) as extract_file:
            cur_book = None
            for vref, verse in zip(vref_file, extract_file):
                if verse != "\n":
                    cur_book = vref.split(" ")[0]
                    cur_chapter = int(vref.split(" ")[1].split(":")[0].strip())
                    verse_counts[cur_book].append(cur_chapter)
            verse_counts = {k: Counter(v) for k, v in verse_counts.items()}

        # Copy the counts to the data frames
        partially_complete_books = []
        for book, chapter_counts in verse_counts.items():
            book_count = sum(chapter_counts.values())
            complete_book_count = sum(complete_verse_counts[book].values())
            verse_counts_df.loc[project_name][book] = book_count
            verse_percentages_df.loc[project_name][book] = 100 * round(book_count / complete_book_count, 3)
            if book_count < complete_book_count and book_count > 0:
                partially_complete_books.append(book)

        if len(partially_complete_books) > 0:
            partially_complete_projects.append(project_name)
            df = pd.DataFrame(
                columns=list(
                    range(1, max([len(complete_verse_counts[book].keys()) for book in partially_complete_books]) + 1)
                )
            )
            df["book"] = partially_complete_books
            df = df.set_index("book")
            for book in partially_complete_books:
                for chapter, complete_count in complete_verse_counts[book].items():
                    df.loc[book][chapter] = 100 * round(verse_counts[book][chapter] / complete_count, 3)
            df.to_csv(partial_books_path / f"{project_name}.csv")

    # Add overall counts
    for book, chapter_counts in complete_verse_counts.items():
        verse_counts_df.loc["complete", book] = sum(chapter_counts.values())

    to_sum = projects_to_process + ["complete"]
    verse_counts_df.loc[to_sum, "Books"] = verse_counts_df.loc[to_sum].apply(
        lambda row: sum([(1 if ele > 0 else 0) for ele in row]), axis=1
    )
    verse_counts_df.loc[to_sum, "Total"] = verse_counts_df.loc[to_sum][OT_CANON + NT_CANON + DT_CANON].sum(axis=1)
    verse_counts_df.loc[to_sum, "OT"] = verse_counts_df.loc[to_sum][OT_CANON].sum(axis=1)
    verse_counts_df.loc[to_sum, "NT"] = verse_counts_df.loc[to_sum][NT_CANON].sum(axis=1)
    verse_counts_df.loc[to_sum, "DT"] = verse_counts_df.loc[to_sum][DT_CANON].sum(axis=1)
    verse_counts_df.fillna(0, inplace=True)

    verse_percentages_df.loc[projects_to_process, "Total"] = 100 * round(
        verse_counts_df.loc[projects_to_process, "Total"] / verse_counts_df.loc["complete", "Total"], 3
    )
    verse_percentages_df.loc[projects_to_process, "OT"] = 100 * round(
        verse_counts_df.loc[projects_to_process, "OT"] / verse_counts_df.loc["complete", "OT"], 3
    )
    verse_percentages_df.loc[projects_to_process, "NT"] = 100 * round(
        verse_counts_df.loc[projects_to_process, "NT"] / verse_counts_df.loc["complete", "NT"], 3
    )
    verse_percentages_df.loc[projects_to_process, "DT"] = 100 * round(
        verse_counts_df.loc[projects_to_process, "DT"] / verse_counts_df.loc["complete", "DT"], 3
    )
    verse_percentages_df.fillna(0, inplace=True)

    if not deutero:
        for project in project_names:
            if verse_counts_df.loc[project]["DT"] > 0:
                dt_books = [col for col in DT_CANON if verse_counts_df.loc[project][col] > 0]
                LOGGER.warning(
                    f"{project} contains text in books {dt_books}. Use --deutero to include counts for these books."
                )

    # Save cache files
    verse_counts_df.sort_index().drop(index="complete").astype(int).to_csv(verse_counts_path)
    verse_percentages_df.sort_index().to_csv(verse_percentages_path)

    # Filter and save to output folder
    if not deutero:
        verse_counts_df = verse_counts_df.drop(columns=DT_CANON + ["DT"])
        verse_percentages_df = verse_percentages_df.drop(columns=DT_CANON + ["DT"])

        to_update = project_names + ["complete"]
        verse_counts_df.loc[to_update, "Books"] = verse_counts_df.loc[to_update][OT_CANON + NT_CANON].apply(
            lambda row: sum([(1 if ele > 0 else 0) for ele in row]), axis=1
        )
        verse_counts_df.loc[to_update, "Total"] = verse_counts_df.loc[to_update][OT_CANON + NT_CANON].sum(axis=1)
        verse_percentages_df.loc[project_names, "Total"] = 100 * round(
            verse_counts_df.loc[project_names, "Total"] / verse_counts_df.loc["complete", "Total"], 3
        )
    verse_counts_df.loc[["complete"] + sorted(project_names)].astype(int).to_csv(output_path / "verse_counts.csv")
    verse_percentages_df.loc[sorted(project_names)].to_csv(output_path / "verse_percentages.csv")

    # Copy over chapter counts for partially complete books
    for project in project_names:
        cache_path = partial_books_path / f"{project}.csv"
        if cache_path.is_file():
            partial_books_out_path = output_path / f"{project}_detailed_percentages.csv"
            df = pd.read_csv(cache_path, index_col="book")

            if project in projects_to_process and project not in partially_complete_projects:
                df = df.drop(df.index)
                df.to_csv(cache_path)
            elif not deutero:
                df = df.drop(index=DT_CANON, errors="ignore")
                df = df.dropna(axis=1, how="all")

            if len(df.index) == 0:
                partial_books_out_path.unlink(missing_ok=True)
            else:
                df.to_csv(partial_books_out_path)

    # Copy new and updated files to the S3 bucket
    if len(projects_to_process) > 0:
        SIL_NLP_ENV.copy_experiment_to_bucket("verse_counts", bucket_files, True)
        SIL_NLP_ENV.copy_experiment_to_bucket(
            "verse_counts/partially_complete_books", [f"{f}.csv" for f in projects_to_process], True
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect various counts from a corpus of Bible extracts")
    parser.add_argument(
        "--input-folder", default=SIL_NLP_ENV.mt_scripture_dir, help="Folder with corpus of Bible extracts"
    )
    parser.add_argument("--output-folder", help="Folder in which to save results", required=True)
    parser.add_argument(
        "--files",
        help="Semicolon-delimited list of patterns of extract file names to count (e.g. 'arb-*.txt;de-NT.txt)",
        required=True,
    )
    parser.add_argument(
        "--deutero",
        default=False,
        action="store_true",
        help="Include counts for Deuterocanon books",
    )
    parser.add_argument("--recount", default=False, action="store_true", help="Force recount of verse counts")
    args = parser.parse_args()

    collect_verse_counts(args.input_folder, args.output_folder, args.files, args.deutero, args.recount)


if __name__ == "__main__":
    main()
