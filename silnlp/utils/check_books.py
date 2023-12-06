import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Union

from lxml import etree

from .. import sfm
from ..sfm import style, usfm

from machine.scripture import VerseRef, book_number_to_id, get_chapters
from ..common.environment import SIL_NLP_ENV
from ..common.paratext import book_file_name_digits, get_project_dir,  get_book_path, get_iso
from ..common.translator import get_stylesheet, collect_segments

LOGGER = logging.getLogger(__package__ + ".translate")

def parse_book(src_project:str, book:str):

    errors = []

    src_project_dir = get_project_dir(src_project)
    #print(src_project_dir)
    

    with (src_project_dir / "Settings.xml").open("rb") as settings_file:
        settings_tree = etree.parse(settings_file)
    
    #src_iso = get_iso(settings_tree)
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

#            if not include_inline_elements:
#                remove_inline_elements(doc)
#
#            segments = collect_segments(book, doc)
#            sentences = [s.text.strip() for s in segments]
#            vrefs = [s.ref for s in segments]

            LOGGER.info(f"{book} in project {src_project} parsed correctly.")
        else:
            LOGGER.info(f"The error above occured while parsing {book} in project {src_project}")
            for error in errors:
                error_str = " ".join([str(s) for s in error.args])
                LOGGER.info(error_str)

def translate_usfm(
    self,
    src_file_path: Path,
    trg_file_path: Path,
    src_iso: str,
    trg_iso: str,
    chapters: List[int] = [],
    trg_project_path: str = "",
    stylesheet: dict = usfm.relaxed_stylesheet,
    include_inline_elements: bool = False,
) -> None:
    with src_file_path.open(mode="r", encoding="utf-8-sig") as book_file:
        doc: List[sfm.Element] = list(usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False))

    book = ""
    for elem in doc:
        if elem.name == "id":
            book = str(elem[0]).strip()[:3]
            break
    if book == "":
        raise RuntimeError(f"The USFM file {src_file_path} doesn't contain an id marker.")

    if not include_inline_elements:
        remove_inline_elements(doc)

    segments = collect_segments(book, doc)

    sentences = [s.text.strip() for s in segments]
    vrefs = [s.ref for s in segments]
    LOGGER.info(f"File {src_file_path} parsed correctly.")

    # Translate select chapters
    if len(chapters) > 0:
        idxs_to_translate = []
        sentences_to_translate = []
        vrefs_to_translate = []
        for i in range(len(sentences)):
            if vrefs[i].chapter_num in chapters:
                idxs_to_translate.append(i)
                sentences_to_translate.append(sentences[i])
                vrefs_to_translate.append(vrefs[i])

        partial_translation = list(self.translate(sentences_to_translate, src_iso, trg_iso, vrefs_to_translate))

        # Get translation from pre-existing target project to fill in translation
        if trg_project_path != "":
            trg_project_book_path = get_book_path(trg_project_path, book)
            if trg_project_book_path.exists():
                with trg_project_book_path.open(mode="r", encoding="utf-8-sig") as book_file:
                    trg_doc: List[sfm.Element] = list(
                        usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False)
                    )
                if not include_inline_elements:
                    remove_inline_elements(trg_doc)
                trg_segments = collect_segments(book, trg_doc)
                trg_sentences = [s.text.strip() for s in trg_segments]
                trg_vrefs = [s.ref for s in trg_segments]

                translations = insert_translation_into_trg_sentences(
                    partial_translation, vrefs_to_translate, trg_sentences, trg_vrefs, chapters
                )
                update_segments(trg_segments, translations)
                with trg_file_path.open(mode="w", encoding="utf-8", newline="\n") as output_file:
                    output_file.write(sfm.generate(trg_doc))
                return

        translations = [""] * len(sentences)
        for i, idx in enumerate(idxs_to_translate):
            translations[idx] = partial_translation[i]
    else:
        translations = list(self.translate(sentences, src_iso, trg_iso, vrefs))

    update_segments(segments, translations)

    with trg_file_path.open(mode="w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(sfm.generate(doc))



def main() -> None:
    parser = argparse.ArgumentParser(description="Translates text using an NMT model")
    parser.add_argument("--src-project", default=None, type=str, help="The source project to translate")
    parser.add_argument(
        "--books", metavar="books", nargs="+", default=[], help="The books to translate; e.g., 'NT', 'OT', 'GEN,EXO'"
    )
    parser.add_argument(
        "--include-inline-elements",
        default=False,
        action="store_true",
        help="Include inline elements for projects in USFM format",
    )
    args = parser.parse_args()

    books = ";".join(args.books)
    book_nums = get_chapters(books)
    books = [book_number_to_id(book) for book in book_nums.keys()]

    for book in books:
        parse_book(src_project = args.src_project, book=book)

if __name__ == "__main__":
    main()
