import argparse
import logging
from pathlib import Path
from typing import List

from lxml import etree
from machine.scripture import book_number_to_id, get_chapters

from .. import sfm
from ..common.paratext import get_book_path, get_project_dir
from ..common.translator import collect_segments, get_stylesheet
from ..sfm import usfm
from .collect_verse_counts import DT_canon, NT_canon, OT_canon
        
valid_canons = ['NT', 'OT', 'DT']
valid_books = []
valid_books.extend(OT_canon)
valid_books.extend(NT_canon)
valid_books.extend(DT_canon)

LOGGER = logging.getLogger(__package__ + ".translate")


def parse_book(src_project_dir: str, book: str):

    errors = []

    # print(src_project_dir)

    #    with (src_project_dir / "Settings.xml").open("rb") as settings_file:
    #        settings_tree = etree.parse(settings_file)

    # src_iso = get_iso(settings_tree)
    book_path = get_book_path(src_project_dir, book)
    stylesheet = get_stylesheet(src_project_dir)

    if not book_path.is_file():
        raise RuntimeError(f"Can't find file {book_path} for book {book}")
    else:
        # LOGGER.info(f"Found the file {book_path} for book {book}")

        with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
            try:
                doc: List[sfm.Element] = list(
                    usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False)
                )
            except Exception as e:
                errors.append(e)

            if not errors:
                book = ""
                for elem in doc:
                    if elem.name == "id":
                        book = str(elem[0]).strip()[:3]
                        break
                if book == "":
                    raise RuntimeError(f"The USFM file {book_path} doesn't contain an id marker.")

                segments = collect_segments(book, doc)
                #            sentences = [s.text.strip() for s in segments]
                vrefs = [s.ref for s in segments]

                LOGGER.info(f"{book} in project {src_project_dir} parsed correctly and contains {len(vrefs)} verses.")
            else:
                LOGGER.info(f"The following error occured while parsing {book} in project {src_project_dir}")
                for error in errors:
                    error_str = " ".join([str(s) for s in error.args])
                    LOGGER.info(error_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translates text using an NMT model")
    parser.add_argument("--src-project", default=None, type=str, help="The source project to translate")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'"
    )
    args = parser.parse_args()
    src_project_dir = get_project_dir(args.src_project)

    sfm_files = [
        file for file in src_project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]
    ]
    books_found = [sfm_file.name[2:5] for sfm_file in sfm_files]
    print(f"Found these books {' '.join([book for book in books_found])} in the project_directory: {src_project_dir}")

    if not sfm_files:
        print(f"No sfm or SFM files found in project folder: {src_project_dir}")
    else:
        books = args.books
        canons_to_add = [canon for canon in books if canon in valid_canons]
        print(f"Canons specified are: {canons_to_add}")

        books_to_check = [book for book in books if book in valid_books]
        print(f"Individual books specified are: {books_to_check}\n")
        
        OT_books_found = [book for book in OT_canon if book in books_found]
        NT_books_found = [book for book in NT_canon if book in books_found]
        DT_books_found = [book for book in DT_canon if book in books_found]
        
        #print(f"OT_books_found are {OT_books_found}")
        #print(f"OT_canon is {OT_canon}")

        for canon_to_add in canons_to_add:
            if canon_to_add == 'OT':
                #print("Adding OT books")
                books_to_check.extend(OT_books_found)
                #print(f"Books_to_check are {books_to_check}.")
            if canon_to_add == 'NT':
                #print("Adding NT books")
                books_to_check.extend(NT_books_found)
            if canon_to_add == 'DT':
                #print("Adding DT books")
                books_to_check.extend(DT_books_found)
        
        print(f"All books to check are: {books_to_check}\n")

        if not books_to_check:
            print(f"No books were specified - will check all books.")
            books_to_check = books_found
        else:
            books_to_check = list(set(books_found).intersection(books_to_check))
            print(f"Of the books specified these were found: {books_to_check}")

        book_nums = get_chapters(books_to_check)
        book_nums_to_check = [book_number_to_id(book) for book in book_nums.keys()]

        for book_num_to_check in book_nums_to_check:
            parse_book(src_project_dir, book_num_to_check)

        invalid_books = [book for book in books if book not in valid_books and book not in valid_canons]
        
        if invalid_books:
            print(f"Books can include a corpus identifier such as NT OT or DT or a book identifier.")
            print(f"Valid book identifiers are: {valid_books}")
            print(f"WARNING: These unknown books were not checked: {invalid_books}")

if __name__ == "__main__":
    main()
