import argparse
import logging
from pathlib import Path
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


def parse_book(project_dir: Path, book: str):

    errors = []

    # print(project_dir)

    #    with (project_dir / "Settings.xml").open("rb") as settings_file:
    #        settings_tree = etree.parse(settings_file)

    # src_iso = get_iso(settings_tree)
    book_path = get_book_path(project_dir, book)
    stylesheet = get_stylesheet(project_dir)

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
                vrefs = [s.ref for s in segments]

                LOGGER.info(f"{book} in project {project_dir} parsed correctly and contains {len(vrefs)} verses.")
            else:
                LOGGER.info(f"The following error occured while parsing {book} in project {project_dir}")
                for error in errors:
                    error_str = " ".join([str(s) for s in error.args])
                    LOGGER.info(error_str)


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="check_books",
        description="Checks sfm files for a project with the same parser as translate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
             Books can include corpora NT OT or DT and individual books.
             Old Testament books are :
             GEN, EXO, LEV, NUM, DEU, JOS, JDG, RUT, 1SA, 2SA, 1KI, 2KI, 1CH, 2CH, EZR, NEH, EST, JOB, PSA, PRO, ECC, 
             SNG, ISA, JER, LAM, EZK, DAN, HOS, JOL, AMO, OBA, JON, MIC, NAM, HAB, ZEP, HAG, ZEC, MAL
             
             New Testament books are :
             MAT, MRK, LUK, JHN, ACT, ROM, 1CO, 2CO, GAL, EPH, PHP, COL, 1TH,
             2TH, 1TI, 2TI, TIT, PHM, HEB, JAS, 1PE, 2PE, 1JN, 2JN, 3JN, JUD, REV

             Deuterocanonical books are:
             TOB, JDT, ESG, WIS, SIR, BAR, LJE, S3Y, SUS, BEL, 1MA, 
             2MA, 3MA, 4MA, 1ES, 2ES, MAN, PS2, ODA, PSS, EZA, JUB, ENO
         """
        ),
    )

    parser.add_argument("project_folder", type=Path, help="A single Paratext project folder to check")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'"
    )

    # parser.print_help()
    projects = list()
    args = parser.parse_args()
    
    project_dir = Path(args.project_folder)
    
    if args.books:
        books = args.books
        selected_books = list()

        canons_to_add = [canon for canon in books if canon in valid_canons]
        #print(f"Canons specified are: {canons_to_add}")

        for canon_to_add in canons_to_add:
            if canon_to_add == "OT":
                # print("Adding OT books")
                selected_books.extend(OT_canon)
                # print(f"Books_to_check are {books_to_check}.")
            if canon_to_add == "NT":
                # print("Adding NT books")
                selected_books.extend(NT_canon)
            if canon_to_add == "DT":
                # print("Adding DT books")
                selected_books.extend(DT_canon)

        selected_books.extend(book for book in books if book not in valid_books and book not in canons_to_add)

        print(f"Books to check are: {selected_books}\n")

    else:
        #print(f"Look for all books.")
        books = valid_books

    print(f"Searching {project_dir} for sfm files.")

    sfm_files = [
        file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]
    ]

    books_found = [sfm_file.name[2:5] for sfm_file in sfm_files]
    books_selected_and_found = [book for book in books_found if book in selected_books]
    
    print(f"\nFound these books {' '.join([book for book in books_selected_and_found])}")

    if not sfm_files:
        print(f"No sfm or SFM files found in project folder: {project_dir}")
        
    # Get list of books to check in usual Biblical order.
    #books_selected_and_found # Should be in Biblical order already.

    for book_selected_and_found in books_selected_and_found:
        parse_book(project_dir, book_selected_and_found)

    other_sfm_files = [book for book in books_found if book not in selected_books]

    if other_sfm_files:
         print(f"These sfm files were not checked: {' '.join(sfm for sfm in other_sfm_files)}")


if __name__ == "__main__":
    main()
