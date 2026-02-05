import textwrap
from pathlib import Path

from machine.corpora import FileParatextProjectSettingsParser

from .collect_verse_counts import DT_CANON, NT_CANON, OT_CANON

VALID_CANONS = ["OT", "NT", "DT"]
VALID_BOOKS = OT_CANON + NT_CANON + DT_CANON + ["FRT", "BAK", "OTH", "INT"]


def add_books_argument(parser):
    parser.add_argument(
        "--books",
        metavar="books",
        nargs="*",
        default=[],
        help="Books to check or process (e.g., GEN EXO NT). Use NT, OT, DT for canons",
    )


def get_epilog():
    return textwrap.dedent(
        """
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
    )


def expand_book_list(books):
    """Parse books argument and expand NT/OT/DT into full book lists"""
    books_to_check = []
    canons_to_add = [canon for canon in books if canon in ["NT", "OT", "DT"]]
    for canon_to_add in canons_to_add:
        if canon_to_add == "OT":
            books_to_check += OT_CANON
        if canon_to_add == "NT":
            books_to_check += NT_CANON
        if canon_to_add == "DT":
            books_to_check += DT_CANON
    books_to_check += [book for book in books if book in VALID_BOOKS]
    return [book for book in VALID_BOOKS if book in set(books_to_check)]


def get_sfm_files_to_process(project_dir, books):

    settings = FileParatextProjectSettingsParser(project_dir).parse()
    sfm_files = []
    for book in books:
        sfm_file = Path(project_dir) / settings.get_book_file_name(book)
        if sfm_file.is_file():
            sfm_files.append(sfm_file)

    return sfm_files


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
