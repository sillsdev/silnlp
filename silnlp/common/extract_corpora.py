# Extracts parallel corpora from Paratext projects.
# This script is dependent on the Machine tool being installed.
# To install:
# 1. Install .NET Core SDK (https://dotnet.microsoft.com/download)
# 2. Run "dotnet tool restore"


import argparse
import logging
from typing import Set

from ..common.environment import MT_SCRIPTURE_DIR, MT_TERMS_DIR, PT_PROJECTS_DIR
from .paratext import extract_project, extract_term_renderings

LOGGER = logging.getLogger(__package__ + ".extract_corpora")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracts text corpora from Paratext projects")
    parser.add_argument("projects", nargs="*", metavar="name", help="Paratext project")
    parser.add_argument(
        "--include", metavar="texts", default="", help="The texts to include; e.g., '*NT*', '*OT*', 'GEN,EXO'"
    )
    parser.add_argument(
        "--exclude", metavar="texts", default="", help="The texts to exclude; e.g., '*NT*', '*OT*', 'GEN,EXO'"
    )
    parser.add_argument("--markers", default=False, action="store_true", help="Include USFM markers")

    parser.add_argument("--clearml", default=False, action="store_true", help="Register Extraction in ClearML")

    args = parser.parse_args()

    projects: Set[str] = set(args.projects)

    if args.clearml:
        import datetime

        from clearml import Task

        Task.init(
            project_name="LangTech_ExtractCorpora", task_name=str(args.projects) + "_" + str(datetime.datetime.now())
        )

    # Which projects have data we can find?
    projects_found: Set[str] = set()
    for project in projects:
        project_path = PT_PROJECTS_DIR / project
        if project_path.is_dir():
            projects_found.add(project)

    # Process the projects that have data and tell the user.
    if len(projects_found) > 0:
        MT_SCRIPTURE_DIR.mkdir(exist_ok=True, parents=True)
        MT_TERMS_DIR.mkdir(exist_ok=True, parents=True)
        for project in projects_found:
            LOGGER.info(f"Extracting {project}...")
            extract_project(project, args.include, args.exclude, args.markers)
            extract_term_renderings(project)
            LOGGER.info("Done.")
    else:
        LOGGER.warning(f"Couldn't find any data to process for any project in {PT_PROJECTS_DIR}.")

    # Tell the user which projects couldn't be found.
    for project in projects:
        if project not in projects_found:
            LOGGER.warning(f"Couldn't find project {project} in {PT_PROJECTS_DIR}.")


if __name__ == "__main__":
    main()
