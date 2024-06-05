import argparse
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict
from ..common.environment import SIL_NLP_ENV

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

    # Write the aggregated data to a new CSV file
    output_file = folder_path / "scores.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        for header, rows in data_by_header.items():
            writer.writerows(rows)
            writer.writerow([])  # Add a blank row to separate different types
        # Write the folder path to the last line of the CSV file
        writer.writerow([folder_path])
    print(f"Wrote scores to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate CSV files in a folder.")
    parser.add_argument("folder", type=Path, help="Path to the folder containing CSV files.")
    args = parser.parse_args()

    folder = Path(args.folder)
    
    if not folder.is_dir():
        folder = Path(SIL_NLP_ENV.mt_experiments_dir) / args.folder 
    
    aggregate_csv(folder)

if __name__ == "__main__":
    main()
