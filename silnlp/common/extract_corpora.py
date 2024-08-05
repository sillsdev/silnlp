import argparse
import logging
from time import sleep
from typing import List, Set

from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef, get_books
from tqdm import tqdm

from ..common.corpus import count_lines
from ..common.count_verses import count_verses
from ..common.environment import SIL_NLP_ENV
from .paratext import check_versification, extract_project, extract_term_renderings, get_project_dir

LOGGER = logging.getLogger(__package__ + ".extract_corpora")


def get_expected_verse_count(include: List[str], exclude: List[str]) -> int:
    include_books_set = get_books(include) if len(include) > 0 else None
    exclude_books_set = get_books(exclude) if len(exclude) > 0 else None

    def filter_lines(verse_ref_str: str) -> bool:
        if include_books_set is None and exclude_books_set is None:
            return True

        vref = VerseRef.from_string(verse_ref_str.strip(), ORIGINAL_VERSIFICATION)
        if exclude_books_set is not None and vref.book_num in exclude_books_set:
            return False

        if include_books_set is not None and vref.book_num in include_books_set:
            return True

        return include_books_set is None

    return count_lines(SIL_NLP_ENV.assets_dir / "vref.txt", filter_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracts text corpora from Paratext projects")
    parser.add_argument("projects", nargs="+", metavar="name", help="Paratext project")
    parser.add_argument(
        "--include", metavar="books", nargs="+", default=[], help="The books to include; e.g., 'NT', 'OT', 'GEN'"
    )
    parser.add_argument(
        "--exclude", metavar="books", nargs="+", default=[], help="The books to exclude; e.g., 'NT', 'OT', 'GEN'"
    )
    parser.add_argument("--markers", default=False, action="store_true", help="Include USFM markers")
    parser.add_argument("--lemmas", default=False, action="store_true", help="Extract lemmas if available")
    parser.add_argument("--project-vrefs", default=False, action="store_true", help="Extract project verse refs")

    parser.add_argument("--clearml", default=False, action="store_true", help="Register Extraction in ClearML")

    args = parser.parse_args()
    project_patterns: Set[str] = set(args.projects)

    project_path = SIL_NLP_ENV.pt_projects_dir
    patterns_without_matching_projects = []

    projects_found = []
    for project_pattern in project_patterns:
        matching_projects = project_path.glob(f"*{project_pattern}*")
        if not matching_projects:
            patterns_without_matching_projects.append(project_pattern)
        else:
            for matching_project in matching_projects:
                project_dir = project_path / matching_project
                if project_dir.is_dir():
                    projects_found.append(matching_project)

    projects_found = set(projects_found)

    print("Found these projects to extract:")
    for project_found in projects_found:
        print(f"{project_found.name}")

    if args.clearml:
        import datetime

        from clearml import Task

        Task.init(
            project_name="LangTech_ExtractCorpora", task_name=str(args.projects) + "_" + str(datetime.datetime.now())
        )

    # Process the projects that have data and tell the user.
    if len(projects_found) > 0:
        expected_verse_count = get_expected_verse_count(args.include, args.exclude)
        SIL_NLP_ENV.mt_scripture_dir.mkdir(exist_ok=True, parents=True)
        SIL_NLP_ENV.mt_terms_dir.mkdir(exist_ok=True, parents=True)
        for project in projects_found:
            LOGGER.info(f"Extracting {project}...")
            project_dir = get_project_dir(project)
            check_versification(project_dir)
            corpus_filename, verse_count = extract_project(
                project_dir,
                SIL_NLP_ENV.mt_scripture_dir,
                args.include,
                args.exclude,
                args.markers,
                args.lemmas,
                args.project_vrefs,
            )
            # check if the number of lines in the file is correct (the same as vref.txt)
            LOGGER.info(f"# of Verses: {verse_count}")
            if verse_count != expected_verse_count:
                LOGGER.error(f"The number of verses is {verse_count}, but should be {expected_verse_count}.")
            terms_count = extract_term_renderings(project_dir, corpus_filename, SIL_NLP_ENV.mt_terms_dir)
            LOGGER.info(f"# of Terms: {terms_count}")
            LOGGER.info("Done.")
    else:
        LOGGER.warning(f"Couldn't find any data to process for any project in {SIL_NLP_ENV.pt_projects_dir}.")

    # Tell the user which projects couldn't be found.
    for pattern_without_matching_projects in patterns_without_matching_projects:
        LOGGER.warning(
            f"Couldn't find any project matching pattern: *{pattern_without_matching_projects}* in {SIL_NLP_ENV.pt_projects_dir}."
        )

    count_verses(SIL_NLP_ENV.mt_scripture_dir, SIL_NLP_ENV.mt_experiments_dir, recount=False)


if __name__ == "__main__":
    main()
