import logging
import string
from abc import ABC, abstractmethod
from datetime import date
from itertools import groupby
from pathlib import Path
from typing import Iterable, List, Optional, Union

import docx
import nltk
from iso639 import Lang
from lxml import etree
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef

from .. import sfm
from ..sfm import style, usfm
from .corpus import load_corpus, write_corpus
from .paratext import get_book_path, get_iso, get_project_dir

LOGGER = logging.getLogger(__package__ + ".translate")
nltk.download("punkt")


class Paragraph:
    def __init__(self, elem: sfm.Element, child_indices: Iterable[int] = [], text: str = ""):
        self.elem = elem
        self.child_indices = list(child_indices)
        self.text = text

    def add_text(self, index: int, text: str) -> None:
        self.child_indices.append(index)
        self.text += text

    def copy(self) -> "Paragraph":
        return Paragraph(self.elem, self.child_indices, self.text)


class Segment:
    def __init__(self, ref: VerseRef, paras: Iterable[Paragraph] = []):
        self.ref = ref
        self.paras = list(paras)

    @property
    def is_empty(self) -> bool:
        return self.text == ""

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.paras)

    def add_text(self, index: int, text: str) -> None:
        if len(text) == 0:
            return
        self.paras[-1].add_text(index, text)

    def reset(self) -> None:
        self.paras.clear()

    def copy(self) -> "Segment":
        return Segment(self.ref.copy(), (p.copy() for p in self.paras if len(p.child_indices) > 0))


def get_char_style_text(elem: sfm.Element) -> str:
    text: str = ""
    for child in elem:
        if isinstance(child, sfm.Element):
            text += get_char_style_text(child)
        elif isinstance(child, sfm.Text):
            text += str(child)
    return text


def get_style(elem: sfm.Element) -> str:
    return elem.name.rstrip(string.digits)


def collect_segments_from_paragraph(segments: List[Segment], cur_elem: sfm.Element, cur_segment: Segment) -> None:
    style = get_style(cur_elem)
    if style != "q" and style != "b":
        if not cur_segment.is_empty:
            segments.append(cur_segment.copy())
        cur_segment.reset()

    cur_segment.paras.append(Paragraph(cur_elem, text="\n" if style == "b" else ""))
    for i, child in enumerate(cur_elem):
        if isinstance(child, sfm.Element):
            if child.name == "v":
                if not cur_segment.is_empty:
                    segments.append(cur_segment.copy())
                cur_segment.reset()
                cur_segment.ref.verse = child.args[0]
                cur_segment.paras.append(Paragraph(cur_elem))
            elif child.meta["StyleType"] == "Character" and child.name != "fig":
                cur_segment.add_text(i, get_char_style_text(child))
        elif isinstance(child, sfm.Text) and cur_elem.name != "id":
            if i > 0 or child != "\n":
                cur_segment.add_text(i, str(child))

    cur_child_segment = Segment(cur_segment.ref.copy())
    cur_child_segment.ref.verse_num = 0
    for child in cur_elem:
        if isinstance(child, sfm.Element) and child.meta["StyleType"] == "Paragraph":
            if child.name == "c":
                cur_child_segment.ref.chapter = child.args[0]
            collect_segments_from_paragraph(segments, child, cur_child_segment)
    if not cur_child_segment.is_empty:
        segments.append(cur_child_segment)


def collect_segments(book: str, doc: List[sfm.Element]) -> List[Segment]:
    segments: List[Segment] = []
    for root in doc:
        cur_segment = Segment(VerseRef(book, 0, 0, ORIGINAL_VERSIFICATION))
        collect_segments_from_paragraph(segments, root, cur_segment)
        if not cur_segment.is_empty:
            segments.append(cur_segment)
    return segments


def update_segments(segments: List[Segment], translations: List[str]) -> None:
    for segment, translation in zip(reversed(segments), reversed(translations)):
        if segment.text.endswith(" "):
            translation += " "

        lines = translation.splitlines()
        if len(lines) == len(segment.paras):
            for para, line in zip(reversed(segment.paras), reversed(lines)):
                if len(para.child_indices) == 0:
                    continue
                for child_index in reversed(para.child_indices):
                    para.elem.pop(child_index)
                if para.text.endswith("\n"):
                    insert_nl_index = para.child_indices[0]
                    for i in range(len(para.child_indices) - 1):
                        if para.child_indices[i] != para.child_indices[i + 1] - 1:
                            insert_nl_index += 1
                    para.elem.insert(insert_nl_index, sfm.Text("\n", parent=para.elem))
                para.elem.insert(para.child_indices[0], sfm.Text(line, parent=para.elem))
        else:
            first_para = next(p for p in segment.paras if len(p.child_indices) > 0)
            for child_index in reversed(first_para.child_indices):
                first_para.elem.pop(child_index)

            insert_nl_index = first_para.child_indices[0]
            for para in reversed(segment.paras[1:]):
                for child_index in reversed(para.child_indices):
                    para.elem.pop(child_index)

                child: Union[sfm.Element, sfm.Text]
                for child in para.elem:
                    child.parent = first_para.elem
                    first_para.elem.insert(first_para.child_indices[0], child)
                    insert_nl_index += 1

                parent: sfm.Element = para.elem.parent
                parent.remove(para.elem)

            if segment.text.endswith("\n"):
                for i in range(len(first_para.child_indices) - 1):
                    if first_para.child_indices[i] != first_para.child_indices[i + 1] - 1:
                        insert_nl_index += 1
                first_para.elem.insert(insert_nl_index, sfm.Text("\n", parent=first_para.elem))

            first_para.elem.insert(first_para.child_indices[0], sfm.Text(translation, parent=first_para.elem))


def get_stylesheet(project_path: Path) -> dict:
    custom_stylesheet_path = project_path / "custom.sty"
    if custom_stylesheet_path.exists():
        with custom_stylesheet_path.open("r", encoding="utf-8-sig") as file:
            custom_stylesheet = style.parse(file)
        return style.update_sheet(usfm.relaxed_stylesheet, custom_stylesheet, ignore_occurs_under=True)
    return usfm.relaxed_stylesheet


def remove_inline_elements_from_element(cur_elem: sfm.Element) -> None:
    inline_idxs = []
    for i, child in enumerate(cur_elem):
        if isinstance(child, sfm.Element):
            if child.meta.get("TextType") == "NoteText" or child.name in ["fm", "rq", "xtSeeAlso"]:
                inline_idxs.append(i)
            else:
                remove_inline_elements_from_element(child)

    for idx in reversed(inline_idxs):
        del cur_elem[idx]


def remove_inline_elements(doc: List[sfm.Element]) -> None:
    for root in doc:
        remove_inline_elements_from_element(root)


def insert_translation_into_trg_sentences(
    sentences: List[str],
    vrefs: List[VerseRef],
    trg_sentences: List[str],
    trg_vrefs: List[VerseRef],
    chapters: List[int],
) -> List[str]:
    ret = [""] * len(trg_sentences)
    translation_idx = 0
    for i in range(len(trg_sentences)):
        if trg_vrefs[i].chapter_num not in chapters:
            ret[i] = trg_sentences[i]
            continue
        # Skip over rest of verse since the whole verse is put into the first entry
        if (
            i > 0
            and trg_vrefs[i].chapter_num == trg_vrefs[i - 1].chapter_num
            and trg_vrefs[i].verse_num == trg_vrefs[i - 1].verse_num
        ):
            continue
        # If translation_idx gets behind, catch up
        while translation_idx < len(sentences) and (
            trg_vrefs[i].chapter_num > vrefs[translation_idx].chapter_num
            or (
                trg_vrefs[i].chapter_num == vrefs[translation_idx].chapter_num
                and trg_vrefs[i].verse_num > vrefs[translation_idx].verse_num
            )
        ):
            translation_idx += 1

        # Put all parts of the translated verse into the first entry for that verse
        while (
            translation_idx < len(sentences)
            and vrefs[translation_idx].chapter_num == trg_vrefs[i].chapter_num
            and vrefs[translation_idx].verse_num == trg_vrefs[i].verse_num
        ):
            if ret[i] != "":
                ret[i] += " "
            ret[i] += sentences[translation_idx]
            translation_idx += 1

    return ret


def insert_draft_remark(
    doc: List[sfm.Element],
    book: str,
    description: str,
    experiment_ckpt_str: str,
) -> List[sfm.Element]:
    remark = f"This draft of {book} was machine translated on {date.today()} from the {description} using model {experiment_ckpt_str}. It should be reviewed and edited carefully.\n"
    rmk_elem = sfm.Element(
        "rem",
        parent=doc[0][0].parent,
        meta={
            "Endmarker": None,
            "StyleType": "Paragraph",
        },
        content=[sfm.Text(remark)],
    )

    doc[0].insert(1, rmk_elem)
    return doc


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
        trg_project: str = "",
        include_inline_elements: bool = False,
        experiment_ckpt_str: str = "",
    ) -> None:
        src_project_dir = get_project_dir(src_project)
        with (src_project_dir / "Settings.xml").open("rb") as settings_file:
            settings_tree = etree.parse(settings_file)
        src_iso = get_iso(settings_tree)
        book_path = get_book_path(src_project, book)
        stylesheet = get_stylesheet(src_project_dir)

        if not book_path.is_file():
            raise RuntimeError(f"Can't find file {book_path} for book {book}")
        else:
            LOGGER.info(f"Found the file {book_path} for book {book}")

        self.translate_usfm(
            book_path,
            output_path,
            src_iso,
            trg_iso,
            chapters,
            trg_project,
            stylesheet,
            include_inline_elements,
            experiment_ckpt_str,
        )

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
        experiment_ckpt_str: str = "",
    ) -> None:
        with src_file_path.open(mode="r", encoding="utf-8-sig") as book_file:
            doc: List[sfm.Element] = list(usfm.parser(book_file, stylesheet=stylesheet, canonicalise_footnotes=False))

        book = ""
        description = ""
        for elem in doc:
            if elem.name == "id":
                doc_str = str(elem[0]).strip()
                book = doc_str[:3]
                if len(doc_str) > 3:
                    description = doc_str[3:].strip(" -")
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
                    trg_doc = insert_draft_remark(trg_doc, book, description, experiment_ckpt_str)

                    with trg_file_path.open(mode="w", encoding="utf-8", newline="\n") as output_file:
                        output_file.write(sfm.generate(trg_doc))

                    return

            translations = [""] * len(sentences)
            for i, idx in enumerate(idxs_to_translate):
                translations[idx] = partial_translation[i]
        else:
            translations = list(self.translate(sentences, src_iso, trg_iso, vrefs))

        update_segments(segments, translations)
        doc = insert_draft_remark(doc, book, description, experiment_ckpt_str)

        with trg_file_path.open(mode="w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(sfm.generate(doc))

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
