import argparse
import logging
import string
import time
from pathlib import Path
from typing import Iterable, List, Optional

logging.basicConfig()

import sentencepiece as sp

from .. import sfm
from ..common.canon import book_id_to_number
from ..common.corpus import load_corpus, write_corpus
from ..common.paratext import get_book_path
from ..common.utils import get_git_revision_hash
from ..sfm import usfm
from .config import create_runner, load_config
from .langs_config import LangsConfig
from .runner import SILRunner
from .utils import decode_sp, decode_sp_lines, encode_sp, encode_sp_lines, get_best_model_dir, get_last_checkpoint


def insert_tag(text: str, trg_iso: Optional[str]) -> str:
    if trg_iso is None:
        return text
    return f"<2{trg_iso}> {text}"


def infer_text_file(
    runner: SILRunner,
    src_spp: sp.SentencePieceProcessor,
    src_paths: List[Path],
    trg_paths: List[Path],
    checkpoint_path: Path,
    step: int,
    trg_iso: Optional[str],
):
    checkpoint_name = "averaged checkpoint" if step == -1 else f"checkpoint {step}"
    print(f"Inferencing {checkpoint_name}...")

    for src_path, trg_path in zip(src_paths, trg_paths):
        start = time.time()

        sentences = list(encode_sp_lines(src_spp, (insert_tag(s, trg_iso) for s in load_corpus(src_path))))
        translations = runner.infer_list(sentences, checkpoint_path=str(checkpoint_path))
        write_corpus(trg_path, decode_sp_lines(map(lambda t: t[0], translations)))

        end = time.time()
        print(f"Inferenced {src_path.name} to {trg_path.name} in {((end-start)/60):.2f} minutes")


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


def infer_book(
    runner: SILRunner,
    src_spp: sp.SentencePieceProcessor,
    src_project: str,
    book: str,
    checkpoint_path: Path,
    output_path: Path,
    trg_iso: Optional[str],
) -> None:
    book_path = get_book_path(src_project, book)
    with open(book_path, mode="r", encoding="utf-8") as book_file:
        doc = list(usfm.parser(book_file))

    segments: List[Segment] = []
    cur_segment = Segment()
    collect_segments(segments, doc[0], cur_segment)
    if not cur_segment.is_empty:
        segments.append(cur_segment)

    features_list: List[str] = list(map(lambda s: encode_sp(src_spp, insert_tag(s.text.strip(), trg_iso)), segments))
    labels_list = runner.infer_list(features_list, str(checkpoint_path))

    for segment, hypotheses in zip(reversed(segments), reversed(labels_list)):
        first_para = segment.paras[0]
        detok_trg_text = decode_sp(hypotheses[0])
        if segment.text.endswith(" "):
            detok_trg_text += " "

        for para in reversed(segment.paras):
            for child_index in reversed(para.child_indices):
                para.elem.pop(child_index)
            if para.add_nl:
                insert_nl_index = para.child_indices[0]
                for i in range(len(para.child_indices) - 1):
                    if para.child_indices[i] != para.child_indices[i + 1] - 1:
                        insert_nl_index += 1
                para.elem.insert(insert_nl_index, sfm.Text("\n", parent=first_para.elem))

        first_para.elem.insert(first_para.child_indices[0], sfm.Text(detok_trg_text, parent=first_para.elem))

    with open(output_path, mode="w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(sfm.generate(doc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests a NMT model using OpenNMT-tf")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to use (last, best, avg, or checkpoint #)")
    parser.add_argument("--src-prefix", default=None, type=str, help="Source file prefix (e.g., de-news2019-)")
    parser.add_argument("--trg-prefix", default=None, type=str, help="Target file prefix (e.g., en-news2019-)")
    parser.add_argument("--start-seq", default=None, type=int, help="Starting file sequence #")
    parser.add_argument("--end-seq", default=None, type=int, help="Ending file sequence #")
    parser.add_argument("--src-project", default=None, type=str, help="The source project to translate")
    parser.add_argument("--book", default=None, type=str, help="The book to translate")
    parser.add_argument("--trg-lang", default=None, type=str, help="The target language to translate into")
    parser.add_argument("--output-usfm", default=None, type=str, help="The output USFM file path")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    exp_name = args.experiment
    config = load_config(exp_name)

    config.set_seed()

    checkpoint: Optional[str] = args.checkpoint
    if checkpoint is not None:
        checkpoint = checkpoint.lower()
    if checkpoint is None or checkpoint == "last":
        checkpoint_path, step = get_last_checkpoint(config.model_dir)
    elif checkpoint == "best":
        best_model_path, step = get_best_model_dir(config.model_dir)
        checkpoint_path = best_model_path / "ckpt"
    elif checkpoint == "avg":
        checkpoint_path, _ = get_last_checkpoint(config.model_dir / "avg")
        step = -1
    else:
        checkpoint_path = config.model_dir / f"ckpt-{checkpoint}"
        step = int(checkpoint)

    runner = create_runner(config, memory_growth=args.memory_growth)

    src_spp = config.create_src_sp_processor()

    trg_iso: Optional[str] = None
    if len(config.trg_isos) > 1:
        trg_iso = args.trg_lang
        if trg_iso is None:
            trg_iso = config.default_trg_iso

    book: Optional[str] = args.book
    if book is not None:
        src_project: Optional[str] = args.src_project
        if src_project is None:
            if not isinstance(config, LangsConfig) or len(config.src_projects) > 1:
                raise RuntimeError("A source project must be specified.")
            src_project = next(iter(config.src_projects))

        step_str = "avg" if step == -1 else str(step)
        default_output_dir = config.exp_dir / "infer" / step_str
        output_path: Optional[Path] = None if args.output_usfm is None else Path(args.output_usfm)
        if output_path is None:
            book_num = book_id_to_number(book)
            output_path = default_output_dir / f"{book_num:02}{book}.SFM"
        elif output_path.name == output_path:
            output_path = default_output_dir / output_path

        output_path.parent.mkdir(exist_ok=True, parents=True)
        infer_book(runner, src_spp, src_project, book, checkpoint_path, output_path, trg_iso)
    elif args.src_prefix is not None:
        if args.trg_prefix is None:
            raise RuntimeError("A target file prefix must be specified.")
        if args.start_seq is None or args.end_seq is None:
            raise RuntimeError("Start and end sequence numbers must be specified.")

        src_file_names: List[Path] = []
        trg_file_names: List[Path] = []
        cwd = Path.cwd()
        for i in range(args.start_seq, args.end_seq + 1):
            file_num = f"{i:04d}"
            src_file_path = cwd / f"{args.src_prefix}{file_num}.txt"
            trg_file_path = cwd / f"{args.trg_prefix}{file_num}.txt"
            if src_file_path.is_file() and not trg_file_path.is_file():
                src_file_names.append(src_file_path)
                trg_file_names.append(trg_file_path)

        infer_text_file(runner, src_spp, src_file_names, trg_file_names, checkpoint_path, step, trg_iso)
    else:
        raise RuntimeError("A Scripture book or source file prefix must be specified.")


if __name__ == "__main__":
    main()
