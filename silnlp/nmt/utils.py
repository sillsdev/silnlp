import os
import glob
from typing import IO, Iterable, Iterator, List, Optional, Tuple, cast
import enum
import re

import opennmt.utils
import sacrebleu
import sentencepiece as sp
import tensorflow as tf


# Different types of parent model checkpoints (last, best, average)
class CheckpointType(enum.Enum):
    last = 1
    best = 2
    average = 3


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
    best_model_path: Optional[str] = None
    step = 0
    for model in sorted(models, key=lambda m: int(m), reverse=True):
        path = os.path.join(export_path, model)
        if os.path.isdir(path):
            best_model_path = path
            step = int(model)
            break
    if best_model_path is None:
        raise RuntimeError("There is no exported models.")
    return best_model_path, step


def get_checkpoint_path(model_dir: str, checkpoint_type: CheckpointType) -> Tuple[str, int]:
    if checkpoint_type == CheckpointType.average:
        # Get the checkpoint path and step count for the averaged checkpoint
        checkpoints = glob.glob(os.path.join(model_dir, "avg", "ckpt-*.index"))
        for checkpoint in sorted(checkpoints, reverse=True):
            if os.path.isfile(checkpoint):
                checkpoint_path = checkpoint
                match = re.search(r"ckpt-(\d+).index", checkpoint_path)
                if match is None:
                    raise RuntimeError("The checkpoint index file is invalid.")
                step = int(match.group(1))
                return checkpoint_path, step
        raise RuntimeError("There is no averaged checkpoint.")
    elif checkpoint_type == CheckpointType.best:
        # Get the checkpoint path and step count for the best checkpoint
        export_path = os.path.join(model_dir, "export")
        models = os.listdir(export_path)
        for model in sorted(models, key=lambda m: int(m), reverse=True):
            path = os.path.join(export_path, model)
            if os.path.isdir(path):
                checkpoint_path = os.path.join(path, "ckpt")
                step = int(model)
                return checkpoint_path, step
        raise RuntimeError("There are no exported models.")
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {checkpoint_type}")


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


@opennmt.utils.register_scorer(name="bleu_multi_ref")
class BLEUMultiRefScorer(opennmt.utils.Scorer):
    def __init__(self):
        super().__init__("bleu")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        with tf.io.gfile.GFile(hyp_path) as sys_stream:
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
                bleu = sacrebleu.corpus_bleu(sys_stream, cast(List[Iterable[str]], refs), force=True)
                return bleu.score
            finally:
                for ref_stream in ref_streams:
                    ref_stream.close()
