import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..common.environment import SIL_NLP_ENV


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
            header = tuple(rows[0])  # Use tuple to make it hashable

            # Add columns to the beginning of each row
            if header not in data_by_header:
                data_by_header[header].append(["Series", "Experiment", "Steps"] + list(header))
            for row in rows[1:]:
                data_by_header[header].append([series, experiment, steps] + row)

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
    with pd.ExcelWriter(output_file) as writer:
        for i, (header, rows) in enumerate(data_by_header.items()):
            # Create a DataFrame for the current header
            df = pd.DataFrame(rows[1:], columns=rows[0])
            # Convert columns to appropriate data types
            df = df.apply(pd.to_numeric, errors="ignore")
            # Generate a unique sheet name
            sheet_name = f"Table_{i + 1}"
            # Write the DataFrame to the Excel file
            df.to_excel(writer, sheet_name=sheet_name, index=False)
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
