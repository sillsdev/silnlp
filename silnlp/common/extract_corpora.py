# Extracts parallel corpora from Paratext projects.
# This script is dependent on the Machine tool being installed.
# To install:
# 1. Install .NET Core SDK (https://dotnet.microsoft.com/download)
# 2. Run "dotnet tool restore"


import argparse
from typing import Set

from ..common.environment import MT_SCRIPTURE_DIR, MT_TERMS_DIR, PT_PROJECTS_DIR
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

    projects: Set[str] = set(args.projects)

    # Which projects have data we can find?
    projects_found: Set[str] = set()
    for project in projects:
        project_path = PT_PROJECTS_DIR / project
        if project_path.is_dir():
            projects_found.add(project)

    # Process the projects that have data and tell the user.
    if len(projects_found) > 0:
        MT_SCRIPTURE_DIR.mkdir(exist_ok=True)
        MT_TERMS_DIR.mkdir(exist_ok=True)
        for project in projects_found:
            print(f"Extracting {project}...")
            extract_project(project, args.include, args.exclude)
            extract_term_renderings(project)
            print("Done.")
    else:
        print("Couldn't find any data to process for any project.")

    # Tell the user which projects couldn't be found.
    for project in projects:
        if project not in projects_found:
            print(f"Couldn't find project {project}")


if __name__ == "__main__":
    main()
