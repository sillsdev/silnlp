import argparse
import logging
import os
import string
import time
from typing import Iterable, List, Optional

logging.basicConfig()

import sentencepiece as sp

from .. import sfm
from ..common.canon import book_id_to_number
from ..common.corpus import load_corpus, write_corpus
from ..common.environment import PT_UNZIPPED_DIR
from ..common.utils import get_git_revision_hash, set_seed
from ..sfm import usfm
from .config import create_runner, get_mt_root_dir, load_config, parse_langs
from .runner import RunnerEx
from .utils import decode_sp, decode_sp_lines, encode_sp, encode_sp_lines, get_best_model_dir, get_last_checkpoint


def insert_tag(text: str, trg_iso: Optional[str]) -> str:
    if trg_iso is None:
        return text
    return f"<2{trg_iso}> {text}"


def infer_text_file(
    runner: RunnerEx,
    src_spp: sp.SentencePieceProcessor,
    srcFiles: List[str],
    trgFiles: List[str],
    checkpoint_path: str,
    step: int,
    trg_iso: Optional[str],
):
    checkpoint_name = "averaged checkpoint" if step == -1 else f"checkpoint {step}"
    print(f"Inferencing {checkpoint_name}...")

    for srcFile, trgFile in zip(srcFiles, trgFiles):
        start = time.time()

        sentences = encode_sp_lines(src_spp, map(lambda s: insert_tag(s, trg_iso), load_corpus(srcFile)))
        translations = runner.infer_list(list(sentences), checkpoint_path=checkpoint_path)
        write_corpus(trgFile, decode_sp_lines(map(lambda t: t[0], translations)))

        end = time.time()
        print(
            f"Inferenced {os.path.basename(srcFile)} to {os.path.basename(trgFile)} in {((end-start)/60):.2f} minutes"
        )


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
    return child


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
    runner: RunnerEx,
    src_spp: sp.SentencePieceProcessor,
    src_project: str,
    book: str,
    checkpoint_path: str,
    output_path: str,
    trg_iso: Optional[str],
) -> None:
    book_num = book_id_to_number(book)
    book_filename = f"{book_num}{book}{src_project}.SFM"
    book_path = os.path.join(PT_UNZIPPED_DIR, src_project, book_filename)
    with open(book_path, mode="r", encoding="utf-8") as book_file:
        doc = list(usfm.parser(book_file))

    segments: List[Segment] = []
    cur_segment = Segment()
    collect_segments(segments, doc[0], cur_segment)
    if not cur_segment.is_empty:
        segments.append(cur_segment)

    features_list: List[str] = list(map(lambda s: encode_sp(src_spp, insert_tag(s.text.strip(), trg_iso)), segments))
    labels_list = runner.infer_list(features_list, checkpoint_path)

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
    root_dir = get_mt_root_dir(exp_name)
    config = load_config(exp_name)
    model_dir: str = config["model_dir"]
    data_config: dict = config["data"]

    set_seed(data_config["seed"])

    checkpoint_path: str
    step: int
    checkpoint: Optional[str] = args.checkpoint
    if checkpoint is not None:
        checkpoint = checkpoint.lower()
    if checkpoint is None or checkpoint == "last":
        checkpoint_path, step = get_last_checkpoint(model_dir)
    elif checkpoint == "best":
        best_model_path, step = get_best_model_dir(model_dir)
        checkpoint_path = os.path.join(best_model_path, "ckpt")
        if not os.path.isfile(checkpoint_path + ".index"):
            checkpoint_path = os.path.join(model_dir, "saved_model.pb")
    elif checkpoint == "avg":
        checkpoint_path, _ = get_last_checkpoint(os.path.join(model_dir, "avg"))
        step = -1
    else:
        checkpoint_path = os.path.join(model_dir, f"ckpt-{checkpoint}")
        step = int(checkpoint)

    runner = create_runner(config, memory_growth=args.memory_growth)

    src_spp = sp.SentencePieceProcessor()
    src_spp.Load(os.path.join(root_dir, "sp.model" if data_config["share_vocab"] else "src-sp.model"))

    trg_iso: Optional[str] = None
    trg_langs = parse_langs(data_config["trg_langs"])
    if len(trg_langs) > 1:
        trg_iso = args.trg_lang
        if trg_iso is None:
            trg_iso = next(iter(trg_langs.keys()))

    if args.book is not None:
        if args.output_usfm is None:
            raise RuntimeError("An output file must be specified.")

        src_project: Optional[str] = args.src_project
        if args.src_project is None:
            src_langs = parse_langs(data_config["src_langs"])
            src_project = next(iter(src_langs.values())).data_files[0].project
        infer_book(runner, src_spp, src_project, args.book, checkpoint_path, args.output_usfm, trg_iso)
    elif args.src_prefix is not None:
        if args.trg_prefix is None:
            raise RuntimeError("A target file prefix must be specified.")
        if args.start_seq is None or args.end_seq is None:
            raise RuntimeError("Start and end sequence numbers must be specified.")

        srcFiles: List[str] = []
        trgFiles: List[str] = []
        cwd = os.getcwd()
        for i in range(args.start_seq, args.end_seq + 1):
            fileNum = f"{i:04d}"
            srcFileName = os.path.join(cwd, f"{args.src_prefix}{fileNum}.txt")
            trgFileName = os.path.join(cwd, f"{args.trg_prefix}{fileNum}.txt")
            if os.path.isfile(srcFileName) and not os.path.isfile(trgFileName):
                srcFiles.append(srcFileName)
                trgFiles.append(trgFileName)

        infer_text_file(runner, src_spp, srcFiles, trgFiles, checkpoint_path, step, trg_iso)
    else:
        raise RuntimeError("A Scripture book or source file prefix must be specified.")


if __name__ == "__main__":
    main()
