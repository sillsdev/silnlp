import argparse
import pandas as pd
from pathlib import Path
import yaml
from ..common.environment import SIL_NLP_ENV

def get_filenames_from_yaml(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    filenames = set()
    for pair in config['data']['corpus_pairs']:
        filenames.update(pair['src'])
        filenames.update(pair['trg'])

    # Add ".txt" extension to each filename
    filenames = [f"{filename}.txt" for filename in filenames]

    return list(filenames)

def get_filenames_from_txt(files_txt_path):
    with open(files_txt_path, "r") as f:
        return [line.strip() for line in f]

def filter_verses(verses_csv_path, filenames, output_csv_path):
    # Read the verses CSV file
    verses_df = pd.read_csv(verses_csv_path)

    # Filter the DataFrame to include only rows with filenames from the list
    filtered_df = verses_df[verses_df.iloc[:, 0].isin(filenames)]

    # Ensure all filenames from the list are included in the final DataFrame
    all_filenames_df = pd.DataFrame({verses_df.columns[0]: filenames})
    result_df = all_filenames_df.merge(filtered_df, how='left', left_on=verses_df.columns[0], right_on=verses_df.columns[0])

    # Set the first column as the index
    result_df.set_index(verses_df.columns[0], inplace=True)
    # Sort results alphabetically by file names ignoring case.
    sorted_df = result_df.reindex(sorted(result_df.index, key=lambda x: x.lower()))

    # Write out the new verses.csv file, overwriting any previous one
    sorted_df.to_csv(output_csv_path)

    # Write out the file as an Excel file
    output_xlsx_path = output_csv_path.with_suffix(".xlsx")
    sorted_df.to_excel(output_xlsx_path)
    print(f"Wrote {len(sorted_df)} verse counts to {output_csv_path} and to {output_xlsx_path}")

def main():
    parser = argparse.ArgumentParser(description="Filter verses.csv based on filenames in files.txt or config.yml.")
    parser.add_argument("folder", type=Path, help="Path to the folder containing files.txt or config.yml.")
    args = parser.parse_args()

    verse_counts_file = SIL_NLP_ENV.mt_experiments_dir / "verses" / "verses.csv"
    
    experiment_folder = SIL_NLP_ENV.mt_experiments_dir / args.folder
    experiment_align_folder = experiment_folder / "align"
    alignment_config_file = experiment_align_folder / "config.yml"
    files_txt_file = experiment_align_folder / "files.txt"

    output_csv_file = experiment_align_folder / "verses.csv"
    
    if alignment_config_file.is_file():
        filenames = get_filenames_from_yaml(alignment_config_file)
    elif files_txt_file.is_file():
        filenames = get_filenames_from_txt(files_txt_file)
    else:
        print(f"Couldn't find either an alignment config file {alignment_config_file} or a list of files in {files_txt_file}.")
        exit()

    filter_verses(verse_counts_file, filenames, output_csv_file)

if __name__ == "__main__":
    main()
