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

LOGGER = logging.getLogger(__package__ + ".translate")


def parse_book(src_project: str, book: str):

    errors = []

    src_project_dir = get_project_dir(src_project)
    # print(src_project_dir)

    #    with (src_project_dir / "Settings.xml").open("rb") as settings_file:
    #        settings_tree = etree.parse(settings_file)

    # src_iso = get_iso(settings_tree)
    book_path = get_book_path(src_project, book)
    stylesheet = get_stylesheet(src_project_dir)

    if not book_path.is_file():
        raise RuntimeError(f"Can't find file {book_path} for book {book}")
    else:
        LOGGER.info(f"Found the file {book_path} for book {book}")

    with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
        try:
            doc: List[sfm.Element] = list(usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False))
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

            LOGGER.info(f"{book} in project {src_project} parsed correctly and contains {len(vrefs)} verses.")
        else:
            LOGGER.info(f"The error above occured while parsing {book} in project {src_project}")
            for error in errors:
                error_str = " ".join([str(s) for s in error.args])
                LOGGER.info(error_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translates text using an NMT model")
    parser.add_argument("--src-project", default=None, type=str, help="The source project to translate")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to translate; e.g., 'NT', 'OT', 'GEN,EXO'"
    )
    args = parser.parse_args()

    books = ";".join(args.books)
    book_nums = get_chapters(books)
    books = [book_number_to_id(book) for book in book_nums.keys()]

    for book in books:
        parse_book(args.src_project, book)


if __name__ == "__main__":
    main()
