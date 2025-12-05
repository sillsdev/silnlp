import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
import openpyxl

import pandas as pd

from ..common.environment import SIL_NLP_ENV

# Columns for current style detection and transformation (match actual input files)
CURRENT_STYLE_COLUMNS = [
    "Series", "Experiment", "Steps", "book", "draft_index", "src_iso", "trg_iso", "num_refs", "references", "sent_len",
    "BLEU", "BLEU_1gram_prec", "BLEU_2gram_prec", "BLEU_3gram_prec", "BLEU_4gram_prec",
    "BLEU_brevity_penalty", "BLEU_total_sys_len", "BLEU_total_ref_len",
    "chrF3", "chrF3+", "chrF3++", "spBLEU", "confidence"
]

# Columns to hide in Excel output
COLUMNS_TO_HIDE = [
    "BLEU_1gram_prec", "BLEU_2gram_prec", "BLEU_3gram_prec", "BLEU_4gram_prec", "BLEU_brevity_penalty",
    "BLEU_total_sys_len", "BLEU_total_ref_len", "chrF3", "chrF3+",
    "book", "draft_index", "num_refs", "references", "sent_len", "spBLEU", "confidence"
]

# Final column order for current style
CURRENT_STYLE_OUTPUT_COLUMNS = [
    "src_iso", "trg_iso", "BLEU", "chrF3++",
    "book", "draft_index", "num_refs", "references", "sent_len", "spBLEU", "confidence"
]

def check_for_lock_file(folder: Path, filename: str, file_type: str):
    """Check for lock files and ask the user to close them then exit."""

    if file_type[0] == ".":
        file_type = file_type[1:]

    if file_type.lower() == "csv":
        lockfile = folder / f".~lock.{filename}.{file_type}#"
    elif file_type.lower() == "xlsx":
        lockfile = folder / f"~${filename}.{file_type}"

    if lockfile.is_file():
        print(f"Found lock file: {lockfile}")
        print(f"Please close {filename}.{file_type} in folder {folder} OR delete the lock file and try again.")
        sys.exit()


def is_current_style(header):
    """Check if the header matches the current style (all columns present, additional columns accepted)."""
    header = [col.strip() for col in header]
    return set(CURRENT_STYLE_COLUMNS).issubset(set(header))


def transform_current_style_rows(header, rows):
    """Remove and reorder columns for current style."""
    # Map column name to index
    col_idx = {col: i for i, col in enumerate(header)}
    # Only keep columns in CURRENT_STYLE_OUTPUT_COLUMNS
    new_header = [col for col in CURRENT_STYLE_OUTPUT_COLUMNS if col in col_idx]
    new_rows = []
    for row in rows:
        if len(row) < len(header):
            continue  # skip incomplete or blank rows
        new_row = [row[col_idx[col]] for col in new_header]
        new_rows.append(new_row)
    return new_header, new_rows


def aggregate_scores(folder):
    # Dictionary to store rows by header type
    data_by_header = defaultdict(list)

    # Iterate over all CSV files in the folder and its subfolders
    csv_files = list(folder.rglob("*/scores-*.csv"))
    for csv_file in csv_files:
        print(csv_file)
    if not csv_files :
        print(f"No scores csv files were found in folder {folder.resolve()}")
        sys.exit(0)

    for csv_file in csv_files:
        series = csv_file.parts[-3]  # Extract series folder name
        experiment = csv_file.parts[-2]  # Extract experiment folder name
        steps = csv_file.stem.split("-")[-1]  # Extract steps from file name

        # Read the CSV file and add new columns
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = list(rows[0])  # Use list for easier manipulation

            # Add columns to the beginning of each row
            print(f"Processing {csv_file}")
            if is_current_style(header):
                # Transform header and rows for current style
                transformed_header, transformed_rows = transform_current_style_rows(header, rows[1:])
                # Add Series, Experiment, Steps to the beginning
                new_header = ["Series", "Experiment", "Steps"] + transformed_header
                if tuple(new_header) not in data_by_header:
                    data_by_header[tuple(new_header)].append(new_header)
                for row in transformed_rows:
                    data_by_header[tuple(new_header)].append([series, experiment, steps] + row)
            else:
                # Old style: keep as is
                if tuple(header) not in data_by_header:
                    data_by_header[tuple(header)].append(["Series", "Experiment", "Steps"] + header)
                for row in rows[1:]:
                    data_by_header[tuple(header)].append([series, experiment, steps] + row)

    return data_by_header


def clean_dataframe(df):
    cleaned_df = df.copy()
    cleaned_df = cleaned_df[~((cleaned_df['trg_iso'] == 'ALL') & (cleaned_df['chrF3++'].isna() | (cleaned_df['chrF3++'] == '')))]
    numeric_cols = ['BLEU', 'BLEU_1gram_prec', 'BLEU_2gram_prec', 'BLEU_3gram_prec', 'BLEU_4gram_prec', 'BLEU_brevity_penalty', 'BLEU_total_sys_len', 'BLEU_total_ref_len', 'chrF3', 'chrF3+', 'chrF3++', 'spBLEU', 'confidence', 'num_refs', 'sent_len', 'draft_index']
    for col in numeric_cols:
        if col in cleaned_df.columns: cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    string_cols = ['Series', 'Experiment', 'Steps', 'book', 'src_iso', 'trg_iso', 'references']
    for col in string_cols:
        if col in cleaned_df.columns: cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    return cleaned_df
    

def merge_all_data(data_by_header):
    all_dfs = []
    for header, rows in data_by_header.items():
        if len(rows) > 1:
            df = pd.DataFrame(rows[1:], columns=rows[0])
            all_dfs.append(df)
    final_columns = CURRENT_STYLE_COLUMNS
    if not all_dfs:
        # Return empty DataFrame with expected columns so downstream code can access them safely
        return pd.DataFrame(columns=final_columns)
    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    for col in final_columns:
        if col not in combined.columns:
            combined[col] = None
    return combined[final_columns]


def sort_dataframe(df, sort_by):
    sort_cols = [col for col, _ in sort_by]
    sort_ascending = [asc for _, asc in sort_by]
    return df.sort_values(by=sort_cols, ascending=sort_ascending, na_position='last')


def write_to_excel(df, folder, output_filename):
    output_file = folder / f"{output_filename}.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Scores', index=False)
    wb = openpyxl.load_workbook(output_file)
    ws = wb['Scores']
    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    for col_idx, col_name in enumerate(header_row, 1):
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        if col_name in COLUMNS_TO_HIDE: ws.column_dimensions[col_letter].hidden = True
        else:
            max_length = len(str(col_name)) if col_name else 0
            for cell in ws[col_letter]:
                if cell.row == 1: continue
                try:
                    cell_length = len(str(cell.value)) if cell.value is not None else 0
                    if cell_length > max_length: max_length = cell_length
                except Exception: pass
            ws.column_dimensions[col_letter].width = max_length + 2
    wb.save(output_file)
    print(f"Wrote scores to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate CSV files in a folder.")
    parser.add_argument("folder", type=Path, help="Path to the folder containing CSV files.")
    parser.add_argument(
        "--output_filename",
        type=str,
        default="scores",
        help="Filename suffix without the '.csv' or '.xlsx'. \
            The folder name is added as a prefix to make it easier to distinguish scores files in search results.",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    base_filename = f"{folder.name}_{args.output_filename}"

    if not folder.is_dir():
        folder = Path(SIL_NLP_ENV.mt_experiments_dir) / args.folder

    # Check for lock files and ask the user to close them.
    check_for_lock_file(folder, base_filename, "xlsx")

    # Aggregate the data from all the scores files.
    data = aggregate_scores(folder)
    combined_df = merge_all_data(data)

    # Clean and sort the data
    clean_df = clean_dataframe(combined_df)
    sorted_df = sort_dataframe(clean_df, sort_by=[("Series", True), ("chrF3++", False), ("BLEU", False)])

    # Write the data to an excel file
    write_to_excel(sorted_df, folder, base_filename)


if __name__ == "__main__":
    main()
