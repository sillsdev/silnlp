import logging
from abc import ABC, abstractmethod
from datetime import date
from itertools import groupby
from pathlib import Path
from typing import Iterable, List, Optional

import docx
import nltk
from iso639 import Lang
from machine.corpora import (
    FileParatextProjectSettingsParser,
    FileParatextProjectTextUpdater,
    UpdateUsfmParserHandler,
    UsfmFileText,
    UsfmStylesheet,
    UsfmTextType,
    parse_usfm,
)
from machine.scripture import VerseRef

from .corpus import load_corpus, write_corpus
from .paratext import get_book_path, get_iso, get_project_dir

LOGGER = logging.getLogger(__package__ + ".translate")
nltk.download("punkt")

NON_NOTE_INLINE_ELEMENTS = ["fm", "rq", "xtSeeAlso"]


def insert_draft_remark(
    usfm: str,
    book: str,
    description: str,
    experiment_ckpt_str: str,
) -> str:
    remark = f"\\rem This draft of {book} was machine translated on {date.today()} from {description} using model {experiment_ckpt_str}. It should be reviewed and edited carefully.\n"

    lines = usfm.split("\n")
    insert_idx = (
        1
        + (len(lines) > 1 and (lines[1].startswith("\\ide") or lines[1].startswith("\\usfm")))
        + (len(lines) > 2 and (lines[2].startswith("\\ide") or lines[2].startswith("\\usfm")))
    )
    lines.insert(insert_idx, remark)
    return "\n".join(lines)


class Translator(ABC):
    @abstractmethod
    def translate(
        self, sentences: Iterable[str], src_iso: str, trg_iso: str, vrefs: Optional[Iterable[VerseRef]] = None
    ) -> Iterable[str]:
        pass

    def translate_text(self, src_file_path: Path, trg_file_path: Path, src_iso: str, trg_iso: str) -> None:
        write_corpus(trg_file_path, self.translate(load_corpus(src_file_path), src_iso, trg_iso))

    def translate_book(
        self,
        src_project: str,
        book: str,
        output_path: Path,
        trg_iso: str,
        chapters: List[int] = [],
        trg_project: Optional[str] = None,
        include_inline_elements: bool = False,
        experiment_ckpt_str: str = "",
    ) -> None:
        book_path = get_book_path(src_project, book)
        if not book_path.is_file():
            raise RuntimeError(f"Can't find file {book_path} for book {book}")
        else:
            LOGGER.info(f"Found the file {book_path} for book {book}")

        self.translate_usfm(
            book_path,
            output_path,
            get_iso(src_project),
            trg_iso,
            chapters,
            trg_project,
            include_inline_elements,
            experiment_ckpt_str,
        )

    def translate_usfm(
        self,
        src_file_path: Path,
        out_path: Path,
        src_iso: str,
        trg_iso: str,
        chapters: List[int] = [],
        trg_project: Optional[str] = None,
        include_inline_elements: bool = False,
        experiment_ckpt_str: str = "",
    ) -> None:
        # Create UsfmFileText object for source
        src_from_project = False
        if str(src_file_path).startswith(str(get_project_dir(""))):
            src_from_project = True
            src_settings = FileParatextProjectSettingsParser(src_file_path.parent).parse()
            src_file_text = UsfmFileText(
                src_settings.stylesheet,
                src_settings.encoding,
                src_settings.get_book_id(src_file_path.name),
                src_file_path,
                src_settings.versification,
                include_all_text=True,
                project=src_settings.name,
            )
        else:
            src_file_text = UsfmFileText("usfm.sty", "utf-8-sig", "", src_file_path, include_all_text=True)

        sentences = [s.text.strip() for s in src_file_text]
        vrefs = [s.ref for s in src_file_text]
        LOGGER.info(f"File {src_file_path} parsed correctly.")

        # Filter sentences
        stylesheet = src_settings.stylesheet if src_from_project else UsfmStylesheet("usfm.sty")
        for i in reversed(range(len(sentences))):
            if len(chapters) > 0 and vrefs[i].chapter_num not in chapters:
                sentences.pop(i)
                vrefs.pop(i)
            elif not include_inline_elements:
                marker = vrefs[i].path[-1].name if len(vrefs[i].path) > 0 else ""
                if stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT or marker in NON_NOTE_INLINE_ELEMENTS:
                    sentences.pop(i)
                    vrefs.pop(i)
        # Set aside empty sentences
        empty_sents = []
        for i in reversed(range(len(sentences))):
            if len(sentences[i]) == 0:
                empty_sents.append((i, sentences.pop(i), vrefs.pop(i)))

        translations = list(self.translate(sentences, src_iso, trg_iso, vrefs))

        # Add empty sentences back in
        for idx, sent, vref in reversed(empty_sents):
            translations.insert(idx, sent)
            vrefs.insert(idx, vref)

        rows = [([ref], translation) for ref, translation in zip(vrefs, translations)]

        # Insert translation into the USFM structure of an existing project
        # If the target project is not the same as the translated file's original project,
        # no verses outside of the ones translated will be overwritten
        use_src_project = trg_project is None and src_from_project
        trg_format_project = src_file_path.parent.name if use_src_project else trg_project
        if trg_format_project is not None:
            dest_project_path = get_project_dir(trg_format_project)
            dest_updater = FileParatextProjectTextUpdater(dest_project_path)
            usfm_out = dest_updater.update_usfm(
                src_file_text.id, rows, strip_all_text=use_src_project, prefer_existing_text=False
            )

            if usfm_out is None:
                raise FileNotFoundError(f"Book {src_file_text.id} does not exist in target project {trg_project}")
        # Insert translation into the USFM structure of an individual file
        else:
            with open(src_file_path, encoding="utf-8-sig") as f:
                usfm = f.read()
            handler = UpdateUsfmParserHandler(rows, vrefs[0].book, strip_all_text=True)
            parse_usfm(usfm, handler)
            usfm_out = handler.get_usfm()

        # Insert draft remark and write to output path
        description = f"project {src_file_text.project}" if src_from_project else f"file {src_file_path.name}"
        usfm_out = insert_draft_remark(usfm_out, vrefs[0].book, description, experiment_ckpt_str)
        encoding = src_settings.encoding if src_from_project else "utf-8"
        with out_path.open("w", encoding=encoding) as f:
            f.write(usfm_out)

    def translate_docx(self, src_file_path: Path, trg_file_path: Path, src_iso: str, trg_iso: str) -> None:
        tokenizer: nltk.tokenize.PunktSentenceTokenizer
        try:
            src_lang = Lang(src_iso)
            tokenizer = nltk.data.load(f"tokenizers/punkt/{src_lang.name.lower()}.pickle")
        except:
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

        with src_file_path.open("rb") as file:
            doc = docx.Document(file)

        sentences: List[str] = []
        paras: List[int] = []

        for i in range(len(doc.paragraphs)):
            for sentence in tokenizer.tokenize(doc.paragraphs[i].text, "test"):
                sentences.append(sentence)
                paras.append(i)

        for para, group in groupby(zip(self.translate(sentences, src_iso, trg_iso), paras), key=lambda t: t[1]):
            text = " ".join(s[0] for s in group)
            doc.paragraphs[para].text = text

        with trg_file_path.open("wb") as file:
            doc.save(file)
