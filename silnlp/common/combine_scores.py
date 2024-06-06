import argparse
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict
from ..common.environment import SIL_NLP_ENV
from ..common.count_verses import check_for_lock_file

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
            # Generate a unique sheet name
            sheet_name = f"Table_{i + 1}"
            # Write the DataFrame to the Excel file
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Wrote scores to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate CSV files in a folder.")
    parser.add_argument("folder", type=Path, help="Path to the folder containing CSV files.")
    parser.add_argument("--output_filename", type=str, default="scores", help="Filename for the results files. Usually the default 'scores' is fine. Don't include the file extension.")
    args = parser.parse_args()

    folder = Path(args.folder)
    output_filename = args.output_filename

    if not folder.is_dir():
        folder = Path(SIL_NLP_ENV.mt_experiments_dir) / args.folder 

    # Check for lock files and ask the user to close them.
    check_for_lock_file(folder, output_filename, "csv")
    check_for_lock_file(folder, output_filename, "xlsx")
        
    data = aggregate_csv(folder)
    
    # Write the aggregated data to a new CSV file
    write_to_csv(data, folder, output_filename)

    # Write the aggregated data to an Excel file
    write_to_excel(data, folder, output_filename)

if __name__ == "__main__":
    main()
