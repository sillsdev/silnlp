import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import yaml
from tqdm import tqdm

from .environment import SilNlpEnv


def show_files(message, files):
    if len(files) > 0:
        print(f"{message}")
        for file in files:
            print(file)
    else:
        print(f"{message} None")


def is_valid_encoding(file: Path, encoding: str = "utf-8") -> bool:
    """
    Checks if a file can be successfully decoded with the specified encoding.
    Returns True if valid, False if it raises a UnicodeDecodeError.
    """
    try:
        # Open in text mode with the specific encoding
        with open(file, "r", encoding=encoding) as f:
            # We must iterate through the file to trigger the decode operation
            # on the data. Just opening it is not enough.
            for _ in f:
                pass
        return True
    except UnicodeDecodeError:
        return False
    except Exception as e:
        # Handle other potential errors (e.g., file not found, permission denied)
        print(f"Could not process file {file}: {e}")
        return False


def get_corpus_paths(corpus_pairs: List[dict], environment: SilNlpEnv) -> List[Path]:
    names = []
    for corpus_pair in corpus_pairs:
        for side in ("src", "trg"):
            corpora = corpus_pair[side]
            if isinstance(corpora, str):
                corpora = corpora.split(",")
            names.extend(corpus.strip() if isinstance(corpus, str) else corpus["name"] for corpus in corpora)
    return sorted(set(environment.get_mt_corpus_path(name) for name in names))


def main() -> None:
    parser = argparse.ArgumentParser(description="Find files referred to in config file and check that they exist.")
    parser.add_argument(
        "folder", help="An experiment folder (typically in MT/experiments) that contains a config.yml file."
    )
    args = parser.parse_args()
    environment = SilNlpEnv.create_standard_environment()

    config_path = environment.mt_experiments_dir / args.folder / "config.yml"
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config is None or len(config.keys()) == 0:
        print("Config file is empty.")
        return

    files = get_corpus_paths(config["data"]["corpus_pairs"], environment)
    existing_files = [file for file in files if file.is_file()]
    missing_files = [file for file in files if file not in existing_files]
    if not missing_files:
        show_files(f"\nAll {len(files)} files were found:", existing_files)
    else:
        show_files(f"There are {len(missing_files)} files missing:", missing_files)
        exit()

    print(f"Checking that all {len(existing_files)} existing files can be read as UTF-8:")
    with ThreadPoolExecutor() as ex:
        futs = {ex.submit(is_valid_encoding, f): f for f in existing_files}
        decode_error_files = [futs[fut] for fut in tqdm(as_completed(futs), total=len(futs)) if not fut.result()]

    if not decode_error_files:
        print(f"All {len(existing_files)} existing files can be decoded as UTF-8.")
    else:
        show_files("\nThese files can't be read with UTF-8 encoding:", decode_error_files)


if __name__ == "__main__":
    main()
