import argparse
import logging
import os
import string
from typing import Iterable, List, Optional

logging.basicConfig()

from .. import sfm
from ..common.canon import book_id_to_number
from ..common.paratext import get_book_path
from ..common.utils import get_git_revision_hash, get_mt_exp_dir
from ..sfm import usfm

from html import unescape
from google.cloud import translate_v2 as translate


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


def collect_segments(segments: List[Segment], cur_elem: sfm.Element, cur_segment: Segment) -> None:
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
            if len(child.strip()) > 0:
                cur_segment.add_text(i, str(child))

    cur_child_segment = Segment()
    for child in cur_elem:
        if isinstance(child, sfm.Element) and child.meta["StyleType"] == "Paragraph":
            collect_segments(segments, child, cur_child_segment)
    if not cur_child_segment.is_empty:
        segments.append(cur_child_segment)


def angle_brackets_to_quotes(s: str) -> str:
    return s.replace('<<', '\"').replace('>>','\"').replace('<', "\'").replace('>',"\'")


def google_infer_book(
    src_project: str,
    book: str,
    output_path: str,
    src_iso: str,
    trg_iso: str,
    replace_angle_brackets: bool,
) -> None:
    book_path = get_book_path(src_project, book)
    with open(book_path, mode="r", encoding="utf-8") as book_file:
        doc = list(usfm.parser(book_file))

    segments: List[Segment] = []
    cur_segment = Segment()
    collect_segments(segments, doc[0], cur_segment)
    if not cur_segment.is_empty:
        segments.append(cur_segment)

    # Do something with the segments
    translate_client = translate.Client()

    for s in segments:
        for para in s.paras:
            for child_index in reversed(para.child_indices):
                child = para.elem[child_index]
                if isinstance(child, sfm.Text):
                    text = child.strip().data
                    if len(text) > 0:
                        if replace_angle_brackets:
                            text = angle_brackets_to_quotes(s.text.strip())
                        results = translate_client.translate(text, source_language=src_iso, target_language=trg_iso)
                        hypothesis = unescape(results['translatedText'])
                        para.elem.pop(child_index)
                        para.elem.insert(child_index, sfm.Text(hypothesis, parent=para.elem))
            if para.add_nl:
                para.elem.insert(child_index+1, sfm.Text("\n", parent=para.elem))

    with open(output_path, mode="w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(sfm.generate(doc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests a NMT model using OpenNMT-tf")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--src-project", default=None, type=str, help="The source project to translate")
    parser.add_argument("--book", default=None, type=str, help="The book to translate")
    parser.add_argument("--src-lang", default=None, type=str, help="ISO-639-1 code for source language (e.g., 'ne')")
    parser.add_argument("--trg-lang", default=None, type=str, help="ISO-639-1 code for target language (e.g., 'en')")
    parser.add_argument("--angle-brackets", default=False, action="store_true", help="Replace angle brackets")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    root_dir = get_mt_exp_dir(args.experiment)
    if args.src_project is None:
        raise RuntimeError("Source project must be specified")
    src_project: Optional[str] = args.src_project
    if args.book is None:
        raise RuntimeError("Book must be specified")
    book = args.book
    if args.src_lang is None:
        raise RuntimeError("Source language code must be specified")
    src_iso = args.src_lang
    if args.trg_lang is None:
        raise RuntimeError("Target language code must be specified")
    trg_iso = args.trg_lang

    # Google Translate gets confused if there are angle brackets ('<', '>', '<<', '>>') in the text to be translated,
    # thinking that they are HTML tag markers.  If the translation uses angle brackets for quotes, substitute quotes
    replace_angle_brackets = args.angle_brackets

    default_output_dir = os.path.join(root_dir, src_project)
    book_num = book_id_to_number(book)
    output_path = os.path.join(default_output_dir, f"{book_num:02}{book}.SFM")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    google_infer_book(src_project, book, output_path, src_iso, trg_iso, replace_angle_brackets)


if __name__ == "__main__":
    main()
