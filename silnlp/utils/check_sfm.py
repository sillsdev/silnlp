import argparse
import textwrap
from typing import List

from .. import sfm
from ..common.paratext import get_book_path, get_project_dir
from ..common.translator import get_stylesheet
from ..sfm import usfm
from .collect_verse_counts import DT_canon, NT_canon, OT_canon

valid_canons = ["NT", "OT", "DT"]
valid_books = []
valid_books.extend(OT_canon)
valid_books.extend(NT_canon)
valid_books.extend(DT_canon)


def get_sfm_files(project_dir):
    return [file for file in project_dir.glob("*") if file.is_file() and file.suffix[1:].lower() in ["sfm", "usfm"]]


def quick_check(book_path, stylesheet):
    with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
        try:
            doc: List[sfm.Element] = list(usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False))
            return False
        except Exception as err:
            return err


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="fix_sfm",
        description="Do a quick check of books in a project.",
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

    parser.add_argument(
        "--project", default=None, type=str, help="A Paratext project folder to check. Check All if none is specified."
    )
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to check; e.g., 'NT', 'OT', 'GEN EXO'"
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Describe the process in more detail.")

    projects = list()
    args = parser.parse_args()

    verbose = args.verbose
    
    if args.project:
        project_dir = get_project_dir(args.project)
        if not project_dir.exists():
            raise RuntimeError(f"Can't find the project folder: '{project_dir}'")
        projects.append(project_dir)
        if verbose:
            print(f"Found {project_dir}")

    else:
        projects_dir = get_project_dir("")
        for project_dir in projects_dir.glob("*"):
            if project_dir.is_dir():
                projects.append(project_dir)
        if verbose:
            print(f"Found {len(projects)} folders in {projects_dir}")

    for project_dir in projects:
        stylesheet = get_stylesheet(project_dir)

        if verbose: print(f"Checking {project_dir}")
        settings_filename = project_dir / "Settings.xml"
        if not settings_filename.is_file():
            if verbose:
                print(f"The project directory does not contain a settings file.")
            continue

        sfm_files = get_sfm_files(project_dir)
        if sfm_files:
            # project = {project_dir: {"sfm_files": sfm_files}}

            books_found = [sfm_file.name[2:5] for sfm_file in sfm_files]
            if verbose:
                print(
                    f"Found these books {' '.join([book for book in books_found])} in the project_directory: {project_dir}"
                )

            if not sfm_files:
                if verbose:
                    print(f"No sfm or usfm files found in project folder: {project_dir}")
            else:
                books = args.books
                canons_to_add = [canon for canon in books if canon in valid_canons]
                # print(f"Canons specified are: {canons_to_add}")

                books_to_check = [book for book in books if book in valid_books]
                # print(f"Individual books specified are: {books_to_check}\n")

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

        # print(f"Books to check are: {books_to_check}\n")

        if not books_to_check:
            books_to_check = books_found
            if verbose:
                print("No books were specified - will check all books.")

        # Get list of books to check in usual Biblical order.
        books_to_check = [book for book in valid_books if book in books_to_check]
        if verbose:
            print(f"Project is {project_dir}")
            print(f"Of the books specified these were found: {' '.join(book for book in books_to_check)}")

        error_count = 0
        for book_to_check in books_to_check:
            error = False
            book_path = get_book_path(project_dir, book_to_check)
            #print()
            with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
                print(book_path, book_file.readline().strip())

            try:
                error = quick_check(book_path, stylesheet)
            except FileNotFoundError as err:
                if verbose: 
                    print(f"{err}")
                continue
            if error:
                error_count += 1
                print(f"Book {book_path} failed to parse. {error}")
        if error_count:
            print(f"In project {project_dir} there are {error_count} books that contain at least one error.\n")
        else:
            print(f"No errors were found in any of the books in project {project_dir}\n")


if __name__ == "__main__":
    main()

# Simple errors that we could fix:
# Repeated markers: ie. "\c \c 1" or "\v\v 1" or "\v \v 1"
# Text after Chapter marker "\c 1 text" or "\c 1\r\ntext"  Could be replaced with "\c 1\r\n\\rem "
# missing verse number after \v
# orphan end marker \{marker}*: no matching opening marker \{marker}
# \v 47 text \47  causes
# \v 2. text or punctuation rather than a space after the verse number.
# \v29 space missing between \v and verse number.
# Pan\kinggan backslash used and second word gets interpreted as an unknown marker.  Replace with a space.
# \s Amg Ikapitung Trumpeta \P    '\P' shouldn't appear at the end of a line, and shouldn't be uppercase - delete it.

# Empty verse marker appears between two verses:
# \v 22 “राजाले त्‍यो नोकरलाई भन्‍यो, ‘दुष्‍ट नोकर, तेरो मुखको वचनले म तलाई दण्‍ड दिनेछु। यदि म जे दिंदैन त्‍यही लिन्‍छु, र जे छर्दैन त्‍यसैको कटनी गर्छु भनेर तैले जानेको थिइस भने,
# \v
# \v 23 मैले दिएको पैँसा बैंकमा किन राखेनस्? राख्थिस् भने, म आएर ब्‍याजसहित पैँसा फिर्ता पाउँनेथिएँ।’

# \S1 shouldn't be uppercase or in the middle of a line.
# \v 24 ꤓꤌꤣ꤭ꤚꤢꤪ ꤢ꤬ ... ꤔꤌꤣ꤬ꤒꤢ꤬ꤟꤢꤩ꤬꤯ \S1 ꤋꤝꤤꤢ꤬ꤗꤤ꤬ꤐꤢ꤬ ꤞꤛꤢꤩ꤭
# Replace with
# \v 24 ꤓꤌꤣ꤭ꤚꤢꤪ ꤢ꤬ ... ꤔꤌꤣ꤬ꤒꤢ꤬ꤟꤢꤩ꤬꤯ 
# \s1 ꤋꤝꤤꤢ꤬ꤗꤤ꤬ꤐꤢ꤬ ꤞꤛꤢꤩ꤭

# Missing space after chapter number '1' - incorrect error message - nothing should follow the chapter number.
# \c 1:

# Missing verse number after \v
# \v “27 अब म राजा भएको मन नपराउनेहरुलाई मेरो छेउमा ल्‍याएर मार’।”

# Space on line before marker:
#  \ss बप्‍तिष्‍मा दिने यूहन्‍ना

# Marker not in stylesheet. Replace \ss with \s
# \ss बप्‍तिष्‍मा दिने यूहन्‍ना