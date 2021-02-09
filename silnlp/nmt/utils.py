import os
from typing import IO, Iterable, Iterator, List, Optional, Tuple, cast

import opennmt.utils
import sacrebleu
import sentencepiece as sp
import tensorflow as tf
import yaml


def decode_sp(line: str) -> str:
    return line.replace(" ", "").replace("\u2581", " ").lstrip()


def decode_sp_lines(lines: Iterable[str]) -> Iterable[str]:
    return map(decode_sp, lines)


def encode_sp(spp: Optional[sp.SentencePieceProcessor], line: str) -> str:
    if spp is None:
        return line
    prefix = ""
    if line.startswith("<2"):
        index = line.index(">")
        prefix = line[0 : index + 2]
        line = line[index + 2 :]
    return prefix + " ".join(spp.EncodeAsPieces(line))


def encode_sp_lines(spp: Optional[sp.SentencePieceProcessor], lines: Iterable[str]) -> Iterator[str]:
    return map(lambda l: encode_sp(spp, l), lines)


def get_best_model_dir(model_dir: str) -> Tuple[str, int]:
    export_path = os.path.join(model_dir, "export")
    models = os.listdir(export_path)
    best_model_dir: Optional[str] = None
    step = 0
    for model in sorted(models, key=lambda m: int(m), reverse=True):
        path = os.path.join(export_path, model)
        if os.path.isdir(path):
            best_model_dir = path
            step = int(model)
            break
    if best_model_dir is None:
        raise RuntimeError("There are no exported models.")
    return best_model_dir, step


def get_last_checkpoint(model_dir: str) -> Tuple[str, int]:
    with open(os.path.join(model_dir, "checkpoint"), "r", encoding="utf-8") as file:
        checkpoint_config = yaml.safe_load(file)
        checkpoint_prefix: str = checkpoint_config["model_checkpoint_path"]
        parts = os.path.basename(checkpoint_prefix).split("-")
        checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
        step = int(parts[-1])
        return (checkpoint_path, step)


@opennmt.utils.register_scorer(name="bleu_sp")
class BLEUSentencepieceScorer(opennmt.utils.Scorer):
    def __init__(self):
        super().__init__("bleu")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        with tf.io.gfile.GFile(ref_path) as ref_stream, tf.io.gfile.GFile(hyp_path) as sys_stream:
            sys = decode_sp_lines(sys_stream)
            ref = decode_sp_lines(ref_stream)
            bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True)
            return bleu.score


def load_ref_streams(ref_path: str) -> List[List[str]]:
    ref_streams: List[IO] = []
    try:
        if ref_path.endswith(".0"):
            prefix = ref_path[:-2]
            i = 0
            while os.path.isfile(f"{prefix}.{i}"):
                ref_streams.append(tf.io.gfile.GFile(f"{prefix}.{i}"))
                i += 1
        else:
            ref_streams.append(tf.io.gfile.GFile(ref_path))
        refs: List[List[str]] = []
        for lines in zip(*ref_streams):
            for ref_index in range(len(ref_streams)):
                ref_line = lines[ref_index].strip()
                if len(refs) == ref_index:
                    refs.append([])
                refs[ref_index].append(ref_line)
        return refs
    finally:
        for ref_stream in ref_streams:
            ref_stream.close()


def load_sys_stream(hyp_path: str) -> List[str]:
    sys_stream = []
    with tf.io.gfile.GFile(hyp_path) as f:
        for line in f:
            sys_stream.append(line.rstrip())
    return sys_stream


@opennmt.utils.register_scorer(name="bleu_multi_ref")
class BLEUMultiRefScorer(opennmt.utils.Scorer):
    def __init__(self):
        super().__init__("bleu")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        ref_streams = load_ref_streams(ref_path)
        sys_stream = load_sys_stream(hyp_path)
        bleu = sacrebleu.corpus_bleu(sys_stream, cast(List[Iterable[str]], ref_streams), force=True)
        return bleu.score
