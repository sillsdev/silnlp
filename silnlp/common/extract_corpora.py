import argparse
import logging
from typing import List, Set

from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef, book_id_to_number, get_books

from ..common.corpus import count_lines
from ..common.environment import SIL_NLP_ENV
from .paratext import extract_project, extract_term_renderings, get_project_dir

LOGGER = logging.getLogger(__package__ + ".extract_corpora")


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
        project_path = SIL_NLP_ENV.pt_projects_dir / project
        if project_path.is_dir():
            projects_found.add(project)

    # Process the projects that have data and tell the user.
    if len(projects_found) > 0:
        include_books: List[str] = [book.strip("*") for book in args.include]
        exclude_books: List[str] = [book.strip("*") for book in args.exclude]
        include_books_set = get_books(include_books) if len(include_books) > 0 else None
        exclude_books_set = get_books(exclude_books) if len(exclude_books) > 0 else None

        def filter_lines(verse_ref_str: str) -> bool:
            if include_books_set is None and exclude_books_set is None:
                return True

            vref = VerseRef.from_string(verse_ref_str.strip(), ORIGINAL_VERSIFICATION)
            if exclude_books_set is not None and vref.book_num in exclude_books_set:
                return False

            if include_books_set is not None and vref.book_num in include_books_set:
                return True

            return include_books_set is None

        expected_verse_count = count_lines(SIL_NLP_ENV.assets_dir / "vref.txt", filter_lines)
        SIL_NLP_ENV.mt_scripture_dir.mkdir(exist_ok=True, parents=True)
        SIL_NLP_ENV.mt_terms_dir.mkdir(exist_ok=True, parents=True)
        for project in projects_found:
            LOGGER.info(f"Extracting {project}...")
            project_dir = get_project_dir(project)
            corpus_filename, verse_count = extract_project(
                project_dir, SIL_NLP_ENV.mt_scripture_dir, include_books, exclude_books, args.markers, args.lemmas
            )
            # check if the number of lines in the file is correct (the same as vref.txt)
            LOGGER.info(f"# of Verses: {verse_count}")
            if verse_count != expected_verse_count:
                LOGGER.error(f"The number of verses is {verse_count}, but should be {expected_verse_count}.")
            terms_count = extract_term_renderings(project_dir, corpus_filename)
            LOGGER.info(f"# of Terms: {terms_count}")
            LOGGER.info("Done.")
    else:
        LOGGER.warning(f"Couldn't find any data to process for any project in {SIL_NLP_ENV.pt_projects_dir}.")

    # Tell the user which projects couldn't be found.
    for project in projects:
        if project not in projects_found:
            LOGGER.warning(f"Couldn't find project {project} in {SIL_NLP_ENV.pt_projects_dir}.")


if __name__ == "__main__":
    main()
