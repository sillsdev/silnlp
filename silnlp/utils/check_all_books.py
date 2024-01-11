import argparse
from csv import register_dialect, DictWriter
import logging
from pathlib import Path
from pprint import pprint
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
#valid_books.extend(DT_canon)

LOGGER = logging.getLogger(__package__ + ".translate")


def get_projects_and_books():

    projects_info = dict()
    projects_dir = get_project_dir("")
    project_dirs = [project_dir for project_dir in projects_dir.glob("*") if project_dir.is_dir()]
    for project_dir in project_dirs[:10]:
        stylesheet = get_stylesheet(project_dir)
        print(f"Checking project {project_dir}")
        sfm_files = get_sfm_files(project_dir)
        if sfm_files:

            books_found = [sfm_file.name[2:5] for sfm_file in sfm_files]
            books_to_check = [book for book in valid_books if book in books_found]
            projects_info[project_dir] = books_to_check

    return projects_info


def get_sfm_files(project_dir):
    return [file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]]


def parse_book(src_project_dir: str, book: str):

    book_path = get_book_path(src_project_dir, book)
    stylesheet = get_stylesheet(src_project_dir)

    if not book_path.is_file():
        return f"Can't find file {book_path}"
    else:
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


def parse_book_quick(book_path, stylesheet):

    with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
        try:
            doc: List[sfm.Element] = list(
                usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False)
            )
        except Exception as e:
            return e
        return True
    
        # book = ""
        # for elem in doc:
        #     if elem.name == "id":
        #         book = str(elem[0]).strip()[:3]
        #         break
        # if book == "":
        #     return f"The file {book_path} doesn't contain an id marker."

        # segments = collect_segments(book, doc)
        # vrefs = [s.ref for s in segments]

        # return len(vrefs)

        
def log(file,text):
    
    print(f"{text}")      
    with open(file, 'a', encoding='utf-8', buffering=2) as f:
        f.write(text.rstrip() + "\n")

def main() -> None:
    error_file = Path("E:\Work\Corpora\PT_project_errors.txt")

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
    register_dialect('default')

    projects_and_books = get_projects_and_books()
    pprint(projects_and_books)
    
    column_headers = [book for book in valid_books]
    column_headers.insert(0,"Project")
    print(column_headers)
    
    #write_csv("E:/Work/Corpora/details.csv" , column_headers , column_headers, overwrite=True)
    #print(f'Wrote summary csv file to {summary_csv_file}')
    exit()

    for project_dir in projects_dir.glob("*"):
        if project_dir.is_dir():
            projects.append(project_dir)

    log(error_file, f"Found {len(projects)} folders in {projects_dir}")

    for project_dir in projects:

        stylesheet = get_stylesheet(project_dir)
        #project_dir = Path("S:\Paratext\projects\BNBT_2023_10_06")
        
        log(error_file, f"Checking project {project_dir}")

        sfm_files = get_sfm_files(project_dir)
        if sfm_files:
            
            books_found = [sfm_file.name[2:5] for sfm_file in sfm_files]
            books_to_check = [book for book in valid_books if book in books_found]
            #books_to_check = ["DEU"]

            #book_nums = get_chapters(books_to_check)
            #book_nums_to_check = [book_number_to_id(book) for book in book_nums.keys()]
            #error_file.write(f"books found are {books_to_check}\n")
            
            for book_to_check in books_to_check:
                book_path = get_book_path(project_dir, book_to_check)
                try:
                    result = parse_book_quick(book_path, stylesheet)
                    log(error_file, f"{book_path}\tParsed correctly. With result {result}")
                except Exception as err:
                    log(error_file, f"{book_path}\tDidn't parse correctly. With error {err}")

if __name__ == "__main__":
    main()
