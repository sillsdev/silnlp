import argparse
import logging
from math import exp
import textwrap
from pathlib import Path

from machine.corpora import FileParatextProjectSettingsParser, UsfmFileText
from machine.scripture import book_number_to_id, get_chapters

from .paratext import get_project_dir
from .book_args import DT_CANON, NT_CANON, OT_CANON, VALID_CANONS, VALID_BOOKS, get_epilog, group_bible_books, add_books_argument, expand_book_list, get_sfm_files_to_process


LOGGER = logging.getLogger(__package__ + ".check_books")

def parse_book(project_dir: str, book_path: Path):
    errors = []

    settings = FileParatextProjectSettingsParser(project_dir).parse()
    LOGGER.info(f"Attempting to parse {book_path}.")
    
    if not book_path.is_file():
        raise RuntimeError(f"Can't find {book_path}")

    try:
        file_text = UsfmFileText(
            settings.stylesheet,
            settings.encoding,
            settings.get_book_id(book_path.name),
            book_path,
            settings.versification,
            include_markers=True,
            include_all_text=True,
            project=settings.name,
        )
    except Exception as e:
        errors.append(e)

    if not errors:
        LOGGER.info(f"{book_path} parsed correctly and contains {file_text.count()} verses.")
    else:
        LOGGER.info(f"The following error occured while parsing {book_path}.")
        for error in errors:
            error_str = " ".join([str(s) for s in error.args])
            LOGGER.info(error_str)


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="check_books",
        description="Checks sfm files for a project with the same parser as translate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_epilog(),
    )

    parser.add_argument("project", type=str, help="The name of the project folder.")
    add_books_argument(parser)  # adds the --books arg

    # parser.print_help()
    args = parser.parse_args()
    project_dir = get_project_dir(args.project)
    books = expand_book_list(args.books)
    books = books if books else VALID_BOOKS
    sfm_files = get_sfm_files_to_process(project_dir, books)

    if not args.books:
        LOGGER.info("No books were specified, will check all books.")
        
    LOGGER.info(f"Will check these books: {books}\n")
  
    for sfm_file in sfm_files:
        parse_book(project_dir, sfm_file)


if __name__ == "__main__":
    main()
