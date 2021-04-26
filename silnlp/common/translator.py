import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional, Union
from xml.etree import ElementTree

from .. import sfm
from ..sfm import usfm
from .corpus import load_corpus, write_corpus
from .paratext import get_book_path, get_iso, get_project_dir


class Paragraph:
    def __init__(self, elem: sfm.Element, child_indices: Iterable[int] = [], add_nl: bool = False):
        self.elem = elem
        self.child_indices = list(child_indices)
        self.add_nl = add_nl

    def copy(self) -> "Paragraph":
        return Paragraph(self.elem, self.child_indices, self.add_nl)


class Segment:
    def __init__(self, paras: Iterable[Paragraph] = [], text: str = ""):
        self.paras = list(paras)
        self.text = text

    @property
    def is_empty(self) -> bool:
        return self.text == ""

    def add_text(self, index: int, text: str) -> None:
        self.paras[-1].child_indices.append(index)
        self.paras[-1].add_nl = text.endswith("\n")
        self.text += text

    def reset(self) -> None:
        self.paras.clear()
        self.text = ""

    def copy(self) -> "Segment":
        return Segment(map(lambda p: p.copy(), filter(lambda p: len(p.child_indices) > 0, self.paras)), self.text)


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
    if style != "q":
        if not cur_segment.is_empty:
            segments.append(cur_segment.copy())
        cur_segment.reset()

    cur_segment.paras.append(Paragraph(cur_elem))
    for i, child in enumerate(cur_elem):
        if isinstance(child, sfm.Element):
            if child.name == "v":
                if not cur_segment.is_empty:
                    segments.append(cur_segment.copy())
                cur_segment.reset()
                cur_segment.paras.append(Paragraph(cur_elem))
            elif child.meta["StyleType"] == "Character" and child.name != "fig":
                cur_segment.add_text(i, get_char_style_text(child))
        elif isinstance(child, sfm.Text) and cur_elem.name != "id":
            if len(child) > 0 and child != "\n":
                cur_segment.add_text(i, str(child))

    cur_child_segment = Segment()
    for child in cur_elem:
        if isinstance(child, sfm.Element) and child.meta["StyleType"] == "Paragraph":
            collect_segments_from_paragraph(segments, child, cur_child_segment)
    if not cur_child_segment.is_empty:
        segments.append(cur_child_segment)


def collect_segments(doc: List[sfm.Element]) -> List[Segment]:
    segments: List[Segment] = []
    cur_segment = Segment()
    collect_segments_from_paragraph(segments, doc[0], cur_segment)
    if not cur_segment.is_empty:
        segments.append(cur_segment)
    return segments


def update_segments(segments: List[Segment], translations: List[str]) -> None:
    for segment, translation in zip(reversed(segments), reversed(translations)):
        first_para = segment.paras[0]
        if segment.text.endswith(" "):
            translation += " "

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
        self, sentences: Iterable[str], src_iso: Optional[str] = None, trg_iso: Optional[str] = None
    ) -> Iterable[str]:
        pass

    def translate_text_file(
        self, src_file_path: Path, trg_file_path: Path, src_iso: Optional[str] = None, trg_iso: Optional[str] = None
    ):
        write_corpus(trg_file_path, self.translate(load_corpus(src_file_path), src_iso=src_iso, trg_iso=trg_iso))

    def translate_book(self, src_project: str, book: str, output_path: Path, trg_iso: Optional[str] = None) -> None:
        src_project_dir = get_project_dir(src_project)
        settings_path = src_project_dir / "Settings.xml"
        settings_tree = ElementTree.parse(settings_path)
        src_iso = get_iso(settings_tree)
        book_path = get_book_path(src_project, book)
        with open(book_path, mode="r", encoding="utf-8") as book_file:
            doc = list(usfm.parser(book_file, stylesheet=usfm.relaxed_stylesheet, canonicalise_footnotes=False))

        segments = collect_segments(doc)

        translations = list(self.translate((s.text.strip() for s in segments), src_iso=src_iso, trg_iso=trg_iso))

        update_segments(segments, translations)

        with open(output_path, mode="w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(sfm.generate(doc))
