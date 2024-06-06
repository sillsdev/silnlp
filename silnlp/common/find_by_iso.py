import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Search for files matching ISO codes in a directory")
    parser.add_argument("--directory", type=str, default="S:/MT/scripture/", help="Directory to search")
    parser.add_argument("iso_codes", type=str, nargs="+", help="List of ISO codes to search for")

    args = parser.parse_args()
    projects_folder = Path("S:\Paratext\projects")
    matching_files = []
    for filename in os.listdir(args.directory):
        if filename.endswith(".txt"):
            iso_code = filename.split("-")[0]
            if iso_code in args.iso_codes:
                matching_files.append(os.path.splitext(filename)[0])  # Remove .txt extension

    if matching_files:
        print("Matching files:")
        for file in matching_files:
            print(f"      - {file}")

        for file in matching_files:
            parts = file.split("-", maxsplit=1)
            if len(parts) > 1:
                iso = parts[0]
                project = parts[1]
                project_dir = projects_folder / project
                print(f"{project} exists: {project_dir.is_dir()}")
            else:
                print(f"Couldn't split {file} on '-'")
    else:
        print("No matching files found.")


if __name__ == "__main__":
    main()
