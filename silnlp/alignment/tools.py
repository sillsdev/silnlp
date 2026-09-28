import platform
import subprocess
from pathlib import Path

from ..common.environment import get_env_path, wsl_path

ATOOLS_PATH = Path(get_env_path("FAST_ALIGN_PATH"), "atools")


def is_atools_available() -> bool:
    return ATOOLS_PATH.is_file()


def execute_atools(forward_align_path: Path, reverse_align_path: Path, output_path: Path, sym_heuristic: str) -> None:
    if not is_atools_available():
        raise RuntimeError("atools is not installed.")

    if platform.system() == "Windows":
        args = [
            "wsl",
            wsl_path(ATOOLS_PATH),
            "-i",
            wsl_path(forward_align_path),
            "-j",
            wsl_path(reverse_align_path),
        ]
    else:
        args = [str(ATOOLS_PATH), "-i", str(forward_align_path), "-j", str(reverse_align_path)]
    args.extend(["-c", sym_heuristic])

    with output_path.open("w") as output_file:
        subprocess.run(args, stdout=output_file, stderr=subprocess.DEVNULL)


FAST_ALIGN_PATH = Path(get_env_path("FAST_ALIGN_PATH"), "fast_align")


def is_fast_align_available() -> bool:
    return FAST_ALIGN_PATH.is_file()


def execute_fast_align(input_path: Path, output_path: Path, prob_table_path: Path, reverse: bool) -> None:
    if not is_fast_align_available():
        raise RuntimeError("fast_align is not installed.")

    if platform.system() == "Windows":
        args = ["wsl", wsl_path(FAST_ALIGN_PATH), "-i", wsl_path(input_path), "-p", wsl_path(prob_table_path)]
    else:
        args = [str(FAST_ALIGN_PATH), "-i", str(input_path), "-p", str(prob_table_path)]
    args.extend(["-d", "-o", "-v", "-t", "-18"])
    if reverse:
        args.append("-r")

    with output_path.open("w") as output_file:
        subprocess.run(args, stdout=output_file, stderr=subprocess.DEVNULL)
