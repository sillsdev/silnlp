import argparse
from pathlib import Path
import pickle
from typing import List

def file_names_from_disk(folder):
    print(f"Fetching file names from folder {folder}")
    return [file.name for file in folder.glob("*.txt") if file.is_file() and '-' in file.name]

def update_cache(cache, cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

def read_cache(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}

def get_filenames(folder, cache_file):
    cache = read_cache(cache_file)
    if folder not in cache:
        cache[folder] = file_names_from_disk(folder)
        update_cache(cache, cache_file)
    return cache[folder]

def main() -> None:
    parser = argparse.ArgumentParser(
        prog='find_by_iso',
        description="Looks in the specified, or default directory for files in the given languages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--folder", default="S:/MT/scripture/", type=Path, help="The folder to search.")
    parser.add_argument(
        "isocodes", metavar="isos", nargs="+", default=[], help="The iso codes to look for in the folder."
    )

    args = parser.parse_args()
    folder = Path(args.folder)
    print(f"The folder is {folder}")

    cache_file = Path("find_by_iso.pkl")
    filenames = get_filenames(folder, cache_file)
    iso_filenames = [filename for filename in filenames if filename.split('-')[0] in args.isocodes]
    for iso_filename in iso_filenames:
        print(iso_filename)

if __name__ == "__main__":
    main()
