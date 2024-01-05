import argparse
import logging

# from pathlib import Path
import textwrap
from typing import List

from lxml import etree
from machine.scripture import book_number_to_id, get_chapters

from .. import sfm
from ..common.paratext import get_book_path, get_project_dir
from ..common.translator import collect_segments, get_stylesheet
from ..sfm import usfm
from .collect_verse_counts import DT_canon, NT_canon, OT_canon

valid_canons = ["NT", "OT", "DT"]
valid_books = []
valid_books.extend(OT_canon)
valid_books.extend(NT_canon)
valid_books.extend(DT_canon)

LOGGER = logging.getLogger(__package__ + ".translate")


def get_sfm_files(project_dir):
    return [file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]]


def parse_book(src_project_dir: str, book: str):

    book_path = get_book_path(src_project_dir, book)
    stylesheet = get_stylesheet(src_project_dir)

    if not book_path.is_file():
        return f"Can't find file {book_path}"
    else:
        # LOGGER.info(f"Found the file {book_path} for book {book}")

        with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
            try:
                doc: List[sfm.Element] = list(
                    usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False)
                )
            except Exception as e:
                return e

            book = ""
            for elem in doc:
                if elem.name == "id":
                    book = str(elem[0]).strip()[:3]
                    break
            if book == "":
                return f"The file {book_path} doesn't contain an id marker."

            segments = collect_segments(book, doc)
            vrefs = [s.ref for s in segments]

            return len(vrefs)


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="check_books",
        description="Checks sfm files for a project with the same parser as translate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # parser.add_argument(
    #     "--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'"
    # )

    # parser.print_help()
    projects = list()
    args = parser.parse_args()

    projects_dir = get_project_dir("")
    for project_dir in projects_dir.glob("*"):
        if project_dir.is_dir():
            projects.append(project_dir)

    print(f"Found {len(projects)} folders in {projects_dir}")

    errors = list()
    for project_dir in projects[:10]:
        print(f"Checking {project_dir}")

        sfm_files = get_sfm_files(project_dir)
        if sfm_files:
            project = {}
            books_found = [sfm_file.name[2:5] for sfm_file in sfm_files]
            books_to_check = [book for book in valid_books if book in books_found]

            # book_nums = get_chapters(books_to_check)
            # book_nums_to_check = [book_number_to_id(book) for book in book_nums.keys()]
            print(f"books found are {books_to_check}")

            for book_to_check in books_to_check:
                try:
                    result = parse_book(project_dir, book_to_check)
                except RuntimeError as err:
                    result = f"{err}"
                project[book_to_check] = result

                #print(result, type(result))
                if type(result) is not int:
                    errors.append(f"{book_to_check}  :  {result}\n")
                    print(book_to_check, result)
        else:
            print(f"No sfm files found in {project_dir}")

    with open("E:\Work\Corpora\PT_project_errors.txt", 'w', encoding='utf-8') as error_file:
        error_file.writelines(errors)


if __name__ == "__main__":
    main()
