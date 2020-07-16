import os
import subprocess
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import opennmt.utils
import sacrebleu
import sentencepiece as sp
import tensorflow as tf


def decode_sp(line: str) -> str:
    return line.replace(" ", "").replace("\u2581", " ").lstrip()


def decode_sp_lines(lines: Iterable[str]) -> Iterable[str]:
    return map(decode_sp, lines)


def encode_sp(spp: sp.SentencePieceProcessor, line: str) -> str:
    prefix = ""
    if line.startswith("<2"):
        index = line.index(">")
        prefix = line[0 : index + 2]
        line = line[index + 2 :]
    return prefix + " ".join(spp.EncodeAsPieces(line))


def encode_sp_lines(spp: sp.SentencePieceProcessor, lines: Iterable[str]) -> Iterator[str]:
    return map(lambda l: encode_sp(spp, l), lines)


def get_best_model_dir(config: dict) -> Tuple[str, int]:
    export_path = os.path.join(config["model_dir"], "export")
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


def get_git_revision_hash() -> str:
    script_path = Path(__file__)
    repo_dir = script_path.parent.parent.parent
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=10", "HEAD"], encoding="utf-8"
    ).strip()


@opennmt.utils.register_scorer(name="bleu_sp")
class BLEUSentencepieceScorer(opennmt.utils.Scorer):
    def __init__(self):
        super(BLEUSentencepieceScorer, self).__init__("bleu")

    def __call__(self, ref_path: str, hyp_path: str) -> float:
        with tf.io.gfile.GFile(ref_path) as ref_stream, tf.io.gfile.GFile(hyp_path) as sys_stream:
            sys = decode_sp_lines(sys_stream)
            ref = decode_sp_lines(ref_stream)
            bleu = sacrebleu.corpus_bleu(sys, [ref], lowercase=True)
            return bleu.score


@opennmt.utils.register_scorer(name="bleu_multi_ref")
class BLEUMultiRefScorer(opennmt.utils.Scorer):
    def __init__(self):
        super(BLEUMultiRefScorer, self).__init__("bleu")

    def __call__(self, ref_path, hyp_path):
        with tf.io.gfile.GFile(ref_path) as ref_stream, tf.io.gfile.GFile(hyp_path) as sys_stream:
            refs: List[List[str]] = []
            for line in ref_stream:
                i = 0
                for sentence in line.split(" ||| "):
                    while len(refs) <= i:
                        refs.append([])
                    refs[i].append(sentence.strip())
                    i += 1
            bleu = sacrebleu.corpus_bleu(sys_stream, refs, force=True)
            return bleu.score
