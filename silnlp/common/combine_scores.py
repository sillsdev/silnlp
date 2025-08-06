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
    "book", "draft_index", "src_iso", "trg_iso", "num_refs", "references", "sent_len",
    "BLEU", "BLEU_1gram_prec", "BLEU_2gram_prec", "BLEU_3gram_prec", "BLEU_4gram_prec",
    "BLEU_brevity_penalty", "BLEU_total_sys_len", "BLEU_total_ref_len",
    "chrF3", "chrF3+", "chrF3++", "spBLEU", "confidence"
]

# Columns to filter out
COLUMNS_TO_REMOVE = [
    "BLEU_1gram_prec", "BLEU_2gram_prec", "BLEU_3gram_prec", "BLEU_4gram_prec", "BLEU_brevity_penalty",
    "BLEU_total_sys_len", "BLEU_total_ref_len", "chrF3", "chrF3+"
]

# Columns to move to the end of the row for csv output and hide in Excel output.
COLUMNS_TO_END = [
    "book", "draft_index", "num_refs", "references", "sent_len", "spBLEU", "confidence"
]

# Final column order for current style
CURRENT_STYLE_OUTPUT_COLUMNS = [
    "src_iso", "trg_iso", "BLEU", "chrF3++"
] + COLUMNS_TO_END


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
    """Check if the header matches the current style (all columns present, no extras)."""
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


def aggregate_csv(folder_path):
    # Dictionary to store rows by header type
    data_by_header = defaultdict(list)

    # Iterate over all CSV files in the folder and its subfolders
    for csv_file in folder_path.rglob("*/scores-*.csv"):
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
            #print(f"Header: {header}")
            #print(f"Is current style: {is_current_style(header)}")
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


def write_to_csv(data_by_header, folder, output_filename):

    output_file = folder / f"{output_filename}.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        for header, rows in data_by_header.items():
            writer.writerows(rows)
            writer.writerow([])  # Add a blank row to separate different types
        # Write the folder path to the last line of the CSV file
        writer.writerow([folder])
    print(f"Wrote scores to {output_file}")


def write_to_excel(data_by_header, folder, output_filename):
    output_file = folder / f"{output_filename}.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        sheet_names = []
        for i, (header, rows) in enumerate(data_by_header.items()):
            # Create a DataFrame for the current header
            df = pd.DataFrame(rows[1:], columns=rows[0])
            # Convert columns to appropriate data types
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass
            # Generate a unique sheet name
            sheet_name = f"Table_{i + 1}"
            sheet_names.append(sheet_name)
            # Write the DataFrame to the Excel file
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    # Now, hide columns in COLUMNS_TO_END in all sheets and auto-size visible columns
    wb = openpyxl.load_workbook(output_file)
    for sheet_name in sheet_names:
        ws = wb[sheet_name]
        header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
        # Hide columns in COLUMNS_TO_END
        for col_idx, col_name in enumerate(header_row, 1):
            col_letter = openpyxl.utils.get_column_letter(col_idx)
            if col_name in COLUMNS_TO_END:
                ws.column_dimensions[col_letter].hidden = True
            else:
                # Auto-size visible columns
                max_length = len(str(col_name)) if col_name else 0
                for cell in ws[openpyxl.utils.get_column_letter(col_idx)]:
                    if cell.row == 1:
                        continue  # skip header, already counted
                    try:
                        cell_length = len(str(cell.value)) if cell.value is not None else 0
                        if cell_length > max_length:
                            max_length = cell_length
                    except Exception:
                        pass
                # Add a little extra space
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
    check_for_lock_file(folder, base_filename, "csv")
    check_for_lock_file(folder, base_filename, "xlsx")

    data = aggregate_csv(folder)

    # Write the aggregated data to a new CSV file
    write_to_csv(data, folder, base_filename)

    # Write the aggregated data to an Excel file
    write_to_excel(data, folder, base_filename)


if __name__ == "__main__":
    main()
