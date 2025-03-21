import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import psutil

from ..common.corpus import write_corpus

METEOR_FULLY_SUPPORTED_LANGS = {"en", "cz", "de", "es", "fr", "ar"}


def compute_meteor_score(lang: str, hyps: List[str], refs: List[List[str]]) -> Optional[float]:
    if lang.lower() not in METEOR_FULLY_SUPPORTED_LANGS:
        return None

    meteor_path = Path(os.getenv("METEOR_PATH", "."), "meteor-1.5.jar")
    if not meteor_path.is_file():
        raise RuntimeError("METEOR is not installed.")

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        hyps_path = temp_dir / "hyps.txt"
        refs_path = temp_dir / "refs.txt"

        write_corpus(hyps_path, hyps)
        write_corpus(refs_path, (line for r in zip(*refs) for line in r))

        mem = "2G"
        mem_available_G = psutil.virtual_memory().available / 1e9
        if mem_available_G < 2:
            mem = "1G"

        args: List[str] = [
            "java",
            "-jar",
            f"-Xmx{mem}",
            str(meteor_path),
            str(hyps_path),
            str(refs_path),
            "-l",
            lang,
            "-norm",
            "-r",
            f"{len(refs)}",
            "-q",
        ]
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        result = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

        return float(result.stdout.strip()) * 100
