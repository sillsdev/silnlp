import argparse
import logging
from pathlib import Path
from typing import List
import sys

from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef, get_books

from .corpus import count_lines
from .paratext import check_versification, extract_project, extract_term_renderings
from machine.corpora import FileParatextProjectSettingsParser
from ..common.environment import SIL_NLP_ENV

LOGGER = logging.getLogger(__package__ + ".bulk_extract_local")
SETTINGS_FILENAME = "Settings.xml"


def parse_settings(project):
    settings_file_path = project / SETTINGS_FILENAME
    if not settings_file_path.is_file():
        LOGGER.warning(f"Warning: {SETTINGS_FILENAME} not found.")
        return

    try:
        parser = FileParatextProjectSettingsParser(str(project))
        project_settings = parser.parse()

        # project_settings.name
        # project_settings.full_name
        # if project_settings.encoding:
        #     self.setting_encoding = getattr(project_settings.encoding, 'name', str(project_settings.encoding))

        # if project_settings.versification:
        #     setting_versification = getattr(project_settings.versification, 'name', str(project_settings.versification))
       
            # project_settings.file_name_prefix
            # project_settings.file_name_form
            # project_settings.file_name_suffix
            # project_settings.biblical_terms_list_type
            # project_settings.biblical_terms_project_name
            # project_settings.biblical_terms_file_name
            # project_settings.language_code

    except Exception as e:
        print(f"Error parsing {SETTINGS_FILENAME}: {e}")
        return None

    return project_settings

def get_expected_verse_count(project: Path, include: List[str], exclude: List[str]) -> int:
    include_books_set = get_books(include) if len(include) > 0 else None
    exclude_books_set = get_books(exclude) if len(exclude) > 0 else None
    project_settings = parse_settings(project)

    if project_settings.versification:
        setting_versification = getattr(project_settings.versification, 'name', str(project_settings.versification))
    print(f"Found versification {setting_versification} in {SETTINGS_FILENAME} for {project}")

    def filter_lines(verse_ref_str: str) -> bool:
        if include_books_set is None and exclude_books_set is None:
            return True

        vref = VerseRef.from_string(verse_ref_str.strip(), setting_versification)
        if exclude_books_set is not None and vref.book_num in exclude_books_set:
            return False

        if include_books_set is not None and vref.book_num in include_books_set:
            return True

        return include_books_set is None

    return count_lines(SIL_NLP_ENV.assets_dir / "vref.txt", filter_lines)


def has_settings_file(project_folder: Path) -> bool:
    return (project_folder / SETTINGS_FILENAME).is_file() or (project_folder / SETTINGS_FILENAME.lower()).is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracts text corpora from Paratext projects")
    parser.add_argument("input", type=str, help="The input folder.")
    parser.add_argument("output", type=str, help="The output corpus folder.")
    parser.add_argument("--terms", type=str, required=True, help="The output terms folder.")
    parser.add_argument(
        "--include", metavar="books", nargs="+", default=[], help="The books to include; e.g., 'NT', 'OT', 'GEN'"
    )
    parser.add_argument(
        "--exclude", metavar="books", nargs="+", default=[], help="The books to exclude; e.g., 'NT', 'OT', 'GEN'"
    )
    parser.add_argument("--markers", default=False, action="store_true", help="Include USFM markers")
    parser.add_argument("--lemmas", default=False, action="store_true", help="Extract lemmas if available")
    parser.add_argument("--project-vrefs", default=False, action="store_true", help="Extract project verse refs")

    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    terms_path = Path(args.terms)

    if not input_path.is_dir():
        print(f"Error: Projects folder not found: {args.input}")
        sys.exit(1)

    if not output_path.is_dir():
        print(f"Error: Output folder not found: {args.output}")
        sys.exit(1)

    if not terms_path.is_dir():
        print(f"Error: Output terms folder not found: {args.terms}")
        sys.exit(1)

    # Which folders have a Settings.xml file we can find?
    projects = [folder for folder in input_path.glob("*") if folder.is_dir() and has_settings_file(folder)]

    # Process the projects that have data and tell the user.
    if len(projects) > 0:
        for project in projects:
            LOGGER.info(f"Extracting {project} to {output_path}")
            expected_verse_count = get_expected_verse_count(project, args.include, args.exclude)

            check_versification(project)
            corpus_filename, verse_count = extract_project(
                project,
                output_path,
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
            terms_count = extract_term_renderings(project, corpus_filename, terms_path)
            LOGGER.info(f"# of Terms: {terms_count}")
            LOGGER.info("Done.")
    else:
        LOGGER.warning(f"Couldn't find any data to process for any project in {input_path}.")

if __name__ == "__main__":
    main()
