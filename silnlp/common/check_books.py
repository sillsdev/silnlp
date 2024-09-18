import argparse
import logging
import textwrap
from pathlib import Path

from machine.corpora import FileParatextProjectSettingsParser, UsfmFileText
from machine.scripture import book_number_to_id, get_chapters

from ..common.paratext import get_project_dir
from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON

LOGGER = logging.getLogger(__package__ + ".check_books")

valid_canons = ["NT", "OT", "DT"]
valid_books = []
valid_books.extend(OT_CANON)
valid_books.extend(NT_CANON)
valid_books.extend(DT_CANON)


def group_bible_books(books_found):
    ot_books = set(OT_CANON)
    nt_books = set(NT_CANON)
    dt_books = set(DT_CANON)

    books_set = set(books_found)

    grouped_books = []

    if ot_books.issubset(books_set):
        grouped_books.append("OT")
        books_set -= ot_books

    if nt_books.issubset(books_set):
        grouped_books.append("NT")
        books_set -= nt_books

    if dt_books.issubset(books_set):
        grouped_books.append("DT")
        books_set -= dt_books

    # Add any remaining individual books
    grouped_books.extend(sorted(books_set))

    return grouped_books


def parse_book(project_dir: str, book: str):
    errors = []

    settings = FileParatextProjectSettingsParser(project_dir).parse()
    book_path = Path(project_dir) / settings.get_book_file_name(book)

    if not book_path.is_file():
        raise RuntimeError(f"Can't find file {book_path} for book {book}")

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
        LOGGER.info(f"{book} in project {project_dir} parsed correctly and contains {file_text.count()} verses.")
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

    parser.add_argument("project", type=str, help="The name of the project folder.")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'"
    )

    # parser.print_help()
    args = parser.parse_args()

    project_dir = get_project_dir(args.project)

    sfm_files = [
        file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]
    ]

    books_found = [sfm_file.name[2:5] for sfm_file in sfm_files]

    # Explicitly remove OXX from the list of books to check until it can be dealt with better
    ignore_books = ["0XX"]
    books_found = [book for book in books_found if book not in ignore_books]

    grouped_result = group_bible_books(books_found)
    LOGGER.info(f"Found these books in the project_directory: {' '.join(grouped_result)}")

    if not sfm_files:
        LOGGER.info(f"No sfm or SFM files found in project folder: {project_dir}")
    else:
        books = args.books
        canons_to_add = [canon for canon in books if canon in valid_canons]

        books_to_check = [book for book in books]

        OT_books_found = [book for book in OT_CANON if book in books_found]
        NT_books_found = [book for book in NT_CANON if book in books_found]
        DT_books_found = [book for book in DT_CANON if book in books_found]

        for canon_to_add in canons_to_add:
            if canon_to_add == "OT":
                books_to_check.extend(OT_books_found)
            if canon_to_add == "NT":
                books_to_check.extend(NT_books_found)
            if canon_to_add == "DT":
                books_to_check.extend(DT_books_found)

        LOGGER.info(f"All books to check are: {books_to_check}\n")

        if not books_to_check:
            LOGGER.info("No books were specified, will check all books.")
            books_to_check = books_found
        else:
            LOGGER.info(f"Of the books specified these were found: {books_to_check}")

        book_nums = get_chapters(books_to_check)
        book_nums_to_check = [book_number_to_id(book) for book in book_nums.keys()]

        for book_num_to_check in book_nums_to_check:
            parse_book(project_dir, book_num_to_check)


if __name__ == "__main__":
    main()
