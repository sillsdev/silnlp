import argparse
from pathlib import Path
from numpy import sort
import yaml
from .utils import get_mt_exp_dir
from .environment import SIL_NLP_ENV

exp_dir = SIL_NLP_ENV.mt_experiments_dir
scripture_dir = SIL_NLP_ENV.mt_scripture_dir

def show_files(message, files):
    if files:
        print(message)
        for file in files:
            print(file)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Find files referred to in config file and check that they exist.")
    parser.add_argument(
        "folder",
        help="An experiment folder (typically in MT/experiments) that contains a config.yml file."
    )
    args = parser.parse_args()
    config_path = exp_dir / args.folder / "config.yml"
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config is None or len(config.keys()) == 0:
        print("Config file is empty.")
        return

    files = list()
    corpus_pairs = config['data']['corpus_pairs']
    for corpus_pair in corpus_pairs:
        src_list = corpus_pair['src'] if isinstance(corpus_pair['src'], list) else [corpus_pair['src']]
        trg_list = corpus_pair['trg'] if isinstance(corpus_pair['trg'], list) else [corpus_pair['trg']]
        files.extend([f"{s}.txt" for s in src_list])
        files.extend([f"{t}.txt" for t in trg_list])

    print(f"Looking for files {files}")
    
    missing_files = list()
    existing_files = list()

    for file in [scripture_dir / f for f in sorted(set(files))]:
        if file.is_file() : existing_files.append(file)
        else: missing_files.append(file)
    
    show_files("\nThese files were found:", existing_files)
    show_files("These files were not found:", missing_files)
    print(f"\nFiles found: {len(existing_files)}, Files missing: {len(missing_files)}")

if __name__ == "__main__":
    main()
