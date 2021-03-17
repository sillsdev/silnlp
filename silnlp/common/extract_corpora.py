# Extracts parallel corpora from Paratext projects.
# This script is dependent on the Machine tool being installed.
# To install:
# 1. Install .NET Core SDK (https://dotnet.microsoft.com/download)
# 2. Run "dotnet tool restore"


import argparse
import os

from ..common.environment import PT_PREPROCESSED_DIR, PT_UNZIPPED_DIR
from .paratext import extract_project, extract_term_renderings


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracts text corpora from Paratext projects")
    parser.add_argument("projects", nargs="*", metavar="name", help="Paratext project")
    parser.add_argument(
        "--include", metavar="texts", default="", help="The texts to include; e.g., '*NT*', '*OT*', 'GEN,EXO'"
    )
    parser.add_argument(
        "--exclude", metavar="texts", default="", help="The texts to exclude; e.g., '*NT*', '*OT*', 'GEN,EXO'"
    )
    args = parser.parse_args()

    projects = set(args.projects)

    # Which projects have data we can find?
    projects_found = []

    data_dir = os.path.join(PT_PREPROCESSED_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    terms_dir = os.path.join(PT_PREPROCESSED_DIR, "terms")
    os.makedirs(terms_dir, exist_ok=True)
    for path in os.listdir(PT_UNZIPPED_DIR):
        if path == "Ref" or (len(projects) > 0 and path not in projects):
            continue
        else:
            projects_found.append(path)

    # Process the projects that have data and tell the user.
    if projects_found:
        for project in projects_found:
            print(f"Extracting {project}...")
            extract_project(project, args.include, args.exclude)
            extract_term_renderings(project)
            print(f"Done.")
    else:
        print("Couldn't find any data to process for any project.")

    # Tell the user which projects couldn't be found.
    for project in projects:
        if project not in projects_found:
            print(f"Couldn't find project {project}")


if __name__ == "__main__":
    main()
