import argparse
import logging
import xml.dom.minidom as minidom
import lxml.etree as etree

from xml.etree.ElementTree import ElementTree, SubElement, Comment, tostring
import textwrap
from typing import List

#from lxml import etree
from machine.scripture import book_number_to_id, get_chapters

from .. import sfm
from .paratext import get_book_path, get_project_dir, parse_project_settings
from .translator import collect_segments, get_stylesheet
from ..sfm import usfm
from .collect_verse_counts import DT_canon, NT_canon, OT_canon

valid_canons = ["NT", "OT", "DT"]
valid_books = []
valid_books.extend(OT_canon)
valid_books.extend(NT_canon)
valid_books.extend(DT_canon)

LOGGER = logging.getLogger(__package__ + ".translate")

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def parse_book(project_dir: str, book: str, stylesheet_field_update: str, verbose: bool):

    """ Check whether a book will parse correctly or not. 
    Return True if it parses with the current settings and False otherwise.
    """
    errors = []

    book_path = get_book_path(project_dir, book)
    stylesheet = get_stylesheet(project_dir, stylesheet_field_update)

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



    #parser.print_help()
    args = parser.parse_args()
    project_dir = get_project_dir(args.project)
    stylesheet_field_update = args.stylesheet_field_update
    verbose = args.verbose

    settings_tree = parse_project_settings(project_dir)
    print(etree.tostring(settings_tree, pretty_print=True).decode())

    
    exit()

    sfm_files = [
        file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]
    ]
    

    books_found = []
    # Not all projects use Numbers in the filenames.
    
    # TODO use the filenaming information from the Settings.xml file to find the right names.
    # There is probably code in SILNLP that knows how to do that already.

    for sfm_file in sfm_files:
        if sfm_file.name[:2].isdigit():
            books_found.append(sfm_file.name[2:5])
        else:
            books_found.append(sfm_file.name[:3])


    #books_found = [ sfm_file.name[2:5] for sfm_file in sfm_files if sfm_file.name[:2].isdigit() else sfm_file.name[:3] ]
    if verbose:
        print(f"Found these books {' '.join([book for book in books_found])} in the project_directory: {project_dir}")

    if not sfm_files:
        print(f"No sfm or SFM files found in project folder: {project_dir}")
    else:
        books = args.books
        canons_to_add = [canon for canon in books if canon in valid_canons]
        if verbose:
            print(f"Canons specified are: {canons_to_add}")

        books_to_check = [book for book in books if book in valid_books]
        if verbose:
            print(f"Individual books specified are: {books_to_check}\n")

        OT_books_found = [book for book in OT_canon if book in books_found]
        NT_books_found = [book for book in NT_canon if book in books_found]
        DT_books_found = [book for book in DT_canon if book in books_found]

        # print(f"OT_books_found are {OT_books_found}")
        # print(f"OT_canon is {OT_canon}")

        for canon_to_add in canons_to_add:
            if canon_to_add == "OT":
                # print("Adding OT books")
                books_to_check.extend(OT_books_found)
                # print(f"Books_to_check are {books_to_check}.")
            if canon_to_add == "NT":
                # print("Adding NT books")
                books_to_check.extend(NT_books_found)
            if canon_to_add == "DT":
                # print("Adding DT books")
                books_to_check.extend(DT_books_found)

        if verbose:
            print(f"All books to check are: {books_to_check}\n")

        if not books_to_check:
            if verbose:
                print(f"No books were specified - will check all books.")
            books_to_check = books_found
        else:
            # Get list of books to check in usual Biblical order.
            books_to_check = [book for book in valid_books if book in books_to_check]
        
        if verbose:
            print(f"Of the books specified these were found: {books_to_check}")

        book_nums = get_chapters(books_to_check)
        books_to_check = [book_number_to_id(book) for book in book_nums.keys()]
        
        parsed = []
        failed_to_parse = []

        for book_to_check in books_to_check:
            if parse_book(project_dir, book_to_check, stylesheet_field_update, verbose):
                parsed.append(book_to_check)
            else:
                failed_to_parse.append(book_to_check)

        invalid_books = [book for book in books if book not in valid_books and book not in valid_canons]

        if invalid_books:
            print(f"\nWARNING: These unknown books were not checked: {invalid_books}")
    
    print(f"\nThese books parsed OK: {' '.join(book for book in parsed)}")
    print(f"These books failed to parse: {' '.join(book for book in failed_to_parse)}")


if __name__ == "__main__":
    main()
