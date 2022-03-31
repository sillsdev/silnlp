import os
import re
from pathlib import Path
from typing import IO, Iterable, Iterator, List, Optional, Tuple, cast

import sacrebleu
import sentencepiece as sp
import tensorflow as tf
import yaml
import numpy as np
from opennmt.utils import Scorer, register_scorer

_TAG_PATTERN = re.compile(r"(<\w+> )+")


def decode_sp(line: str) -> str:
    return line.replace(" ", "").replace("\u2581", " ").lstrip()


def decode_sp_lines(lines: Iterable[str]) -> Iterable[str]:
    return map(decode_sp, lines)


def encode_sp(spp: Optional[sp.SentencePieceProcessor], line: str, add_dummy_prefix: Optional[bool] = True) -> str:
    if spp is None:
        return line
    prefix = ""
    match = _TAG_PATTERN.match(line)
    if match is not None:
        index = match.end(0)
        prefix = line[:index]
        line = line[index:]
    if not add_dummy_prefix:
        line = "\ufffc" + line
    pieces = spp.EncodeAsPieces(line)
    if not add_dummy_prefix:
        pieces = pieces[2:]
    return prefix + " ".join(pieces)


def encode_sp_lines(
    spp: Optional[sp.SentencePieceProcessor], lines: Iterable[str], add_dummy_prefix: Optional[bool] = True
) -> Iterator[str]:
    return (encode_sp(spp, l, add_dummy_prefix=add_dummy_prefix) for l in lines)


def get_best_model_dir(model_dir: Path) -> Tuple[Path, int]:
    export_path = model_dir / "export"
    models = list(d.name for d in export_path.iterdir())
    best_model_dir: Optional[Path] = None
    step = 0
    for model in sorted(models, key=lambda m: int(m), reverse=True):
        path = export_path / model
        if path.is_dir():
            best_model_dir = path
            step = int(model)
            break
    if best_model_dir is None:
        raise RuntimeError("There are no exported models.")
    return best_model_dir, step


def get_last_checkpoint(model_dir: Path) -> Tuple[Path, int]:
    with (model_dir / "checkpoint").open("r", encoding="utf-8") as file:
        checkpoint_config = yaml.safe_load(file)
        checkpoint_prefix = Path(checkpoint_config["model_checkpoint_path"])
        parts = checkpoint_prefix.name.split("-")
        checkpoint_path = model_dir / checkpoint_prefix
        step = int(parts[-1])
        return (checkpoint_path, step)


@register_scorer(name="bleu_sp")
class BLEUSentencepieceScorer(Scorer):
    def __init__(self):
        super().__init__("bleu")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        with tf.io.gfile.GFile(ref_path) as ref_stream, tf.io.gfile.GFile(hyp_path) as sys_stream:
            sys = decode_sp_lines(sys_stream)
            ref = decode_sp_lines(ref_stream)
            bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True)
            return bleu.score


def load_ref_streams(ref_path: str, detok: bool = False) -> List[List[str]]:
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
                if not detok:
                    ref_line = lines[ref_index].strip()
                else:
                    ref_line = decode_sp(lines[ref_index].strip())
                if len(refs) == ref_index:
                    refs.append([])
                refs[ref_index].append(ref_line)
        return refs
    finally:
        for ref_stream in ref_streams:
            ref_stream.close()


def load_sys_stream(hyp_path: str, detok: bool = False) -> List[str]:
    sys_stream = []
    with tf.io.gfile.GFile(hyp_path) as f:
        for line in f:
            if not detok:
                sys_stream.append(line.rstrip())
            else:
                sys_stream.append(decode_sp(line.rstrip()))
    return sys_stream


@register_scorer(name="bleu_multi_ref")
class BLEUMultiRefScorer(Scorer):
    def __init__(self):
        super().__init__("bleu")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        ref_streams = load_ref_streams(ref_path)
        sys_stream = load_sys_stream(hyp_path)
        bleu = sacrebleu.corpus_bleu(sys_stream, cast(List[Iterable[str]], ref_streams), force=True)
        return bleu.score


@register_scorer(name="bleu_multi_ref_detok")
class BLEUMultiRefScorer(Scorer):
    def __init__(self):
        super().__init__("bleu_multi_ref_detok")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        ref_streams = load_ref_streams(ref_path, detok=True)
        sys_stream = load_sys_stream(hyp_path, detok=True)
        bleu = sacrebleu.corpus_bleu(sys_stream, cast(List[Iterable[str]], ref_streams), force=True)
        return bleu.score


@register_scorer(name="chrf3")
class chrF3Scorer(Scorer):
    def __init__(self):
        super().__init__("chrf3")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        ref_streams = load_ref_streams(ref_path)
        sys_stream = load_sys_stream(hyp_path)
        chrf3_score = sacrebleu.corpus_chrf(sys_stream, ref_streams, order=6, beta=3, remove_whitespace=True)
        return np.round(float(chrf3_score.score * 100), 2)


@register_scorer(name="chrf3_detok")
class chrF3DetokScorer(Scorer):
    def __init__(self):
        super().__init__("chrf3_detok")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        ref_streams = load_ref_streams(ref_path, detok=True)
        sys_stream = load_sys_stream(hyp_path, detok=True)
        chrf3_score = sacrebleu.corpus_chrf(sys_stream, ref_streams, order=6, beta=3, remove_whitespace=True)
        return np.round(float(chrf3_score.score * 100), 2)


def enable_memory_growth():
    gpus = tf.config.list_physical_devices(device_type="GPU")
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, enable=True)
