import os
import subprocess
import tempfile
from typing import Iterable, List, Optional

import numpy as np
import psutil
import sacrebleu
from opennmt.utils.wer import wer

from ..common.corpus import write_corpus

METEOR_FULLY_SUPPORTED_LANGS = {"en", "cz", "de", "es", "fr", "ar"}


def compute_ter_score(hyps: Iterable[str], refs: List[Iterable[str]]) -> float:
    result = sacrebleu.corpus_ter(hyps,refs)
    return float(np.round(float(result.score) * 100, 2))


def compute_wer_score(hyps: Iterable[str], refs: Iterable[str]) -> float:
    with tempfile.TemporaryDirectory() as td:
        hyps_path = os.path.join(td, "hyps.txt")
        refs_path = os.path.join(td, "refs.txt")

        write_corpus(hyps_path, hyps)
        write_corpus(refs_path, (line for r in zip(*refs) for line in r))

        try:
            result = wer(hyps_path, refs_path)
        except UnicodeDecodeError:
            print("Unable to compute WER score")
            result = -1
        except ZeroDivisionError:
            print("Cannot divide by zero. Check for empty lines.")
            result = -1

        return float(np.round(float(result) * 100, 2))


def compute_meteor_score(lang: str, hyps: Iterable[str], refs: List[Iterable[str]]) -> Optional[float]:
    if lang.lower() not in METEOR_FULLY_SUPPORTED_LANGS:
        return None

    meteor_path = os.path.join(os.getenv("METEOR_PATH", "."), "meteor-1.5.jar")
    if not os.path.isfile(meteor_path):
        raise RuntimeError("METEOR is not installed.")

    with tempfile.TemporaryDirectory() as td:
        hyps_path = os.path.join(td, "hyps.txt")
        refs_path = os.path.join(td, "refs.txt")

        write_corpus(hyps_path, hyps)
        write_corpus(refs_path, (line for r in zip(*refs) for line in r))

        mem = "2G"
        mem_available_G = psutil.virtual_memory().available / 1e9
        if mem_available_G < 2:
            mem = "1G"

        meteor_cmd = [
            "java",
            "-jar",
            f"-Xmx{mem}",
            meteor_path,
            hyps_path,
            refs_path,
            "-l",
            lang,
            "-norm",
            "-r",
            f"{len(refs)}",
            "-q",
        ]
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        #        result = subprocess.run(meteor_cmd, env=env, capture_output=True, encoding="utf-8")
        result = subprocess.run(meteor_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

        return float(np.round(float(result.stdout.strip()) * 100, 2))
