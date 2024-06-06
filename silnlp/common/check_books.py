import argparse
import logging
import textwrap
from typing import List

from .. import sfm
from ..sfm import usfm
from .collect_verse_counts import DT_canon, NT_canon, OT_canon
from .paratext import get_book_path, get_project_dir, parse_project_settings
from .translator import collect_segments, get_stylesheet

valid_canons = ["NT", "OT", "DT"]
valid_books = []
valid_books.extend(OT_canon)
valid_books.extend(NT_canon)
valid_books.extend(DT_canon)


LOGGER = logging.getLogger(__package__ + ".translate")


def parse_book(project_dir, book, stylesheet_field_update: str, verbose: bool):

    """Check whether a book will parse correctly or not.
    Return True if it parses with the current settings and False otherwise.
    """
    errors = []

    book_path = get_book_path(project_dir, book)
    stylesheet = get_stylesheet(project_dir)

    with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
        try:
            doc: List[sfm.Element] = list(usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False))
        except Exception as e:
            parsed_without_errors = False
            errors.append(e)

        if not errors:
            book = ""
            for elem in doc:
                if elem.name == "id":
                    book = str(elem[0]).strip()[:3]
                    break
            if book == "":
                parsed_without_errors = False
                if verbose:
                    raise RuntimeError(f"The USFM file {book_path} doesn't contain an id marker.")

            segments = collect_segments(book, doc)
            #            sentences = [s.text.strip() for s in segments]
            vrefs = [s.ref for s in segments]
            if verbose:
                LOGGER.info(f"{book} in project {project_dir} parsed correctly and contains {len(vrefs)} verses.")
            parsed_without_errors = True

        else:
            if verbose:
                LOGGER.info(f"The following error occured while parsing {book} in project {project_dir}")
                for error in errors:
                    error_str = " ".join([str(s) for s in error.args])
                    LOGGER.info(error_str)
            parsed_without_errors = False

    return parsed_without_errors


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

    parser.add_argument("project", type=str, help="The name of the Project Folder")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'"
    )

    parser.add_argument(
        "--stylesheet-field-update",
        default="merge",
        type=str,
        help="What to do with the OccursUnder and TextProperties fields of a project's custom stylesheet. Possible values are 'replace', 'merge', and 'ignore'.",
    )

    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Print more messages.",
    )

    # parser.print_help()
    args = parser.parse_args()
    project_dir = get_project_dir(args.project)
    stylesheet_field_update = args.stylesheet_field_update
    verbose = args.verbose

    settings_tree = parse_project_settings(project_dir)
    # print(etree.tostring(settings_tree, pretty_print=True).decode())

    # TODO use the filenaming information from the Settings.xml file to find the right names.
    # There is probably code in SILNLP that knows how to do that already.
    books = args.books

    invalid_books = [book for book in books if book not in valid_books and book not in valid_canons]
    if invalid_books:
        print(f"\nWARNING: These unknown books will not be checked: {invalid_books}")

    if not args.books:
        if verbose:
            print(f"No books were specified - will check all books.")
        books_to_check = valid_books

    else:
        canons_to_add = [canon for canon in books if canon in valid_canons]
        if verbose:
            print(f"Canons specified are: {canons_to_add}")

        # Add individually specified books to the list.
        books_to_check = [book for book in books if book in valid_books]
        if verbose:
            print(f"Individual books specified are: {books_to_check}\n")

        # Add books in canons to the list
        for canon_to_add in canons_to_add:
            if canon_to_add == "OT":
                books_to_check.extend(OT_canon)
            if canon_to_add == "NT":
                books_to_check.extend(NT_canon)
            if canon_to_add == "DT":
                books_to_check.extend(DT_canon)

        # Sort books to check in the usual order.
        books_to_check = [book for book in valid_books if book in books_to_check]

    parsed = []
    failed_to_parse = []
    missing = []

    for book_to_check in books_to_check:
        book_path = get_book_path(project_dir, book_to_check)

        if not book_path.is_file():
            missing.append(f"{book_to_check} {book_path}")
        else:
            if parse_book(project_dir, book_to_check, stylesheet_field_update, verbose):
                parsed.append(f"{book_to_check} {book_path}")
            else:
                failed_to_parse.append(f"{book_to_check} {book_path}")

    for book in parsed:
        print(f"Parsed OK: {book}")
    for book in failed_to_parse:
        print(f"Failed to parse: {book}")
    for book in missing:
        print(f"Couldn't find: {book}")


if __name__ == "__main__":
    main()
