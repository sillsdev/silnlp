import argparse
import logging
from typing import Generator, Iterable, Optional

from google.cloud import translate_v2 as translate
from machine.scripture import VerseRef, book_id_to_number

from .paratext import book_file_name_digits
from .translation_data_structures import SentenceTranslation, SentenceTranslationGroup
from .translator import Translator
from .utils import get_git_revision_hash, get_mt_exp_dir

LOGGER = logging.getLogger((__package__ or "") + ".translate")


class GoogleTranslator(Translator):
    def __init__(self) -> None:
        self._translate_client = translate.Client()

    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        vrefs: Optional[Iterable[VerseRef]] = None,
    ) -> Generator[SentenceTranslationGroup, None, None]:
        if produce_multiple_translations:
            LOGGER.warning("Google Translator does not support --multiple-translations")

        for sentence in sentences:
            if len(sentence) == 0:
                yield SentenceTranslationGroup([SentenceTranslation("", [], [], None)])
            else:
                results = self._translate_client.translate(
                    sentence, source_language=src_iso, target_language=trg_iso, format_="text"
                )
                translation: str = results["translatedText"]
                yield SentenceTranslationGroup([SentenceTranslation(translation, [], [], None)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Translates text using Google Cloud")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--src-project", type=str, help="The source project to translate")
    parser.add_argument("--book", type=str, help="The book to translate")
    parser.add_argument("--trg-lang", default=None, type=str, help="ISO-639-1 code for target language (e.g., 'en')")
    args = parser.parse_args()

    get_git_revision_hash()

    root_dir = get_mt_exp_dir(args.experiment)
    src_project: str = args.src_project
    book: str = args.book
    trg_iso: str = "en" if args.trg_lang is None else args.trg_lang

    default_output_dir = root_dir / src_project
    book_num = book_id_to_number(book)
    output_path = default_output_dir / f"{book_file_name_digits(book_num)}{book}.SFM"
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True)

    translator = GoogleTranslator()
    translator.translate_book(src_project, book, output_path, trg_iso, experiment_ckpt_str=args.experiment)


if __name__ == "__main__":
    main()
