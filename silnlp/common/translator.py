import logging
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from itertools import groupby
from pathlib import Path
from typing import Generator, Iterable, List, Optional

import docx
from machine.corpora import FileParatextProjectSettingsParser, ParatextProjectSettings, UsfmFileText, UsfmStylesheet
from machine.scripture import is_book_id_valid

from silnlp.common.utils import add_tags_to_sentence

from .confidence_files import generate_confidence_files
from .corpus import load_corpus, write_corpus
from .environment import SilNlpEnv
from .paratext import get_book_path, get_iso
from .translation_data_structures import DraftGroup, SentenceTranslationGroup, UsfmTextRowCollection
from .usfm_draft_writer import TranslatedUsfm, UsfmSource
from .utils import NLTKSentenceTokenizer

LOGGER = logging.getLogger((__package__ or "") + ".translate")


class Translator(AbstractContextManager["Translator"], ABC):
    def __init__(self, environment: SilNlpEnv):
        self._environment = environment

    @abstractmethod
    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
    ) -> Generator[SentenceTranslationGroup, None, None]:
        pass

    def translate_text(
        self,
        src_file_path: Path,
        trg_file_path: Path,
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        trg_prefix: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:

        sentences = [add_tags_to_sentence(tags, sentence) for sentence in load_corpus(src_file_path)]
        sentence_translation_groups: List[SentenceTranslationGroup] = list(
            self.translate(sentences, src_iso, trg_iso, produce_multiple_translations)
        )
        draft_set = DraftGroup(sentence_translation_groups)
        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            if produce_multiple_translations:
                trg_draft_file_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
            else:
                trg_draft_file_path = trg_file_path
            write_corpus(trg_draft_file_path, translated_draft.get_all_translations())

            if save_confidences:
                generate_confidence_files(
                    translated_draft,
                    trg_draft_file_path,
                )

    def translate_book(
        self,
        src_project: str,
        book: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        chapters: Optional[List[int]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[TranslatedUsfm]:
        book_path = get_book_path(src_project, book, self._environment)
        if not book_path.is_file():
            raise RuntimeError(f"Can't find file {book_path} for book {book}")
        else:
            LOGGER.info(f"Found the file {book_path} for book {book}")

        return self.translate_usfm(
            book_path,
            get_iso(self._environment.get_paratext_project_dir(src_project)),
            trg_iso,
            produce_multiple_translations,
            chapters,
            tags,
        )

    def translate_usfm(
        self,
        src_file_path: Path,
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        chapters: Optional[List[int]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[TranslatedUsfm]:
        # Create UsfmFileText object for source
        src_from_project = False
        src_settings: Optional[ParatextProjectSettings] = None
        stylesheet = UsfmStylesheet("usfm.sty")
        if str(src_file_path).startswith(str(self._environment.get_paratext_project_dir(""))):
            src_from_project = True
            src_settings = FileParatextProjectSettingsParser(src_file_path.parent).parse()
            stylesheet = src_settings.stylesheet
            book_id = src_settings.get_book_id(src_file_path.name)
            assert book_id is not None

            src_file_text = UsfmFileText(
                src_settings.stylesheet,
                src_settings.encoding,
                book_id,
                src_file_path,
                src_settings.versification,
                include_all_text=True,
                project=src_settings.name,
            )
        else:
            # Guess book ID
            with src_file_path.open(encoding="utf-8-sig") as f:
                book_id = f.read().split()[1].upper()
            if not is_book_id_valid(book_id):
                raise ValueError(f"Book ID not detected: {book_id}")

            src_file_text = UsfmFileText(stylesheet, "utf-8-sig", book_id, src_file_path, include_all_text=True)

        sentences = UsfmTextRowCollection(src_file_text, src_iso, stylesheet, chapters, tags)
        LOGGER.info(f"File {src_file_path} parsed correctly.")
        sentences_to_translate = sentences.get_sentences_for_translation()

        if len(sentences_to_translate) == 0:
            LOGGER.warning(f"No sentences found to translate. Skipping translation for {book_id}.")
            return None

        sentence_translation_groups: List[SentenceTranslationGroup] = list(
            self.translate(
                sentences_to_translate,
                src_iso,
                trg_iso,
                produce_multiple_translations,
            )
        )

        return TranslatedUsfm(
            UsfmSource(src_file_path, src_file_text, src_settings, stylesheet, src_from_project, sentences.get_book()),
            chapters,
            sentences.to_translated_text_row_collection(sentence_translation_groups),
        )

    def translate_docx(
        self,
        src_file_path: Path,
        trg_file_path: Path,
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        tags: Optional[List[str]] = None,
    ) -> None:
        with src_file_path.open("rb") as file:
            doc = docx.Document(file)

        sentences: List[str] = []
        paras: List[int] = []

        for i, paragraph in enumerate(doc.paragraphs):
            for sentence in NLTKSentenceTokenizer.for_iso(src_iso).tokenize(paragraph.text):
                sentences.append(add_tags_to_sentence(tags, sentence))
                paras.append(i)

        draft_set: DraftGroup = DraftGroup(
            list(self.translate(sentences, src_iso, trg_iso, produce_multiple_translations))
        )

        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            for para, group in groupby(zip(translated_draft.get_all_translations(), paras), key=lambda t: t[1]):
                text = " ".join(s[0] for s in group)
                doc.paragraphs[para].text = text

            if produce_multiple_translations:
                trg_draft_file_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
            else:
                trg_draft_file_path = trg_file_path

            with trg_draft_file_path.open("wb") as file:
                doc.save(file)

    def __enter__(self) -> "Translator":
        return self

    def __exit__(
        self, exc_type, exc_val, exc_tb  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    ) -> None:
        pass
