import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Union

from lxml import etree
from machine.scripture import ORIGINAL_VERSIFICATION, VerseRef

from .. import sfm
from ..sfm import usfm
from .corpus import load_corpus, write_corpus
from .paratext import get_book_path, get_iso, get_project_dir


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


class Translator(ABC):
    @abstractmethod
    def translate(
        self,
        sentences: Iterable[Union[str, List[str]]],
        src_iso: str,
        trg_iso: str,
    ) -> Iterable[str]:
        pass

    def translate_text_file(self, src_file_path: Path, trg_file_path: Path, src_iso: str, trg_iso: str):
        write_corpus(trg_file_path, self.translate(load_corpus(src_file_path), src_iso, trg_iso))

    def translate_book(self, src_project: str, book: str, output_path: Path, trg_iso: str) -> None:
        src_project_dir = get_project_dir(src_project)
        with (src_project_dir / "Settings.xml").open("rb") as settings_file:
            settings_tree = etree.parse(settings_file)
        src_iso = get_iso(settings_tree)
        book_path = get_book_path(src_project, book)
        with book_path.open(mode="r", encoding="utf-8-sig") as book_file:
            doc = list(usfm.parser(book_file, stylesheet=usfm.relaxed_stylesheet, canonicalise_footnotes=False))

        segments = collect_segments(book, doc)

        translations = list(
            self.translate(
                ([s.text.strip(), str(s.ref) if s.ref.verse_num != 0 else ""] for s in segments), src_iso, trg_iso
            )
        )

        update_segments(segments, translations)

        with output_path.open(mode="w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(sfm.generate(doc))
