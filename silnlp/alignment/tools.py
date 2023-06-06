import platform
import subprocess
from pathlib import Path
from typing import Tuple

from ..common.environment import get_env_path, wsl_path

EFLOMAL_PATH = Path(get_env_path("EFLOMAL_PATH"), "eflomal")


def is_eflomal_available() -> bool:
    return EFLOMAL_PATH.is_file()


def execute_eflomal(
    source_path: Path,
    target_path: Path,
    forward_links_path: Path,
    reverse_links_path: Path,
    n_iterations: Tuple[int, int, int],
) -> None:
    if not is_eflomal_available():
        raise RuntimeError("eflomal is not installed.")

    if platform.system() == "Windows":
        args = [
            "wsl",
            wsl_path(EFLOMAL_PATH),
            "-s",
            wsl_path(source_path),
            "-t",
            wsl_path(target_path),
            "-f",
            wsl_path(forward_links_path),
            "-r",
            wsl_path(reverse_links_path),
        ]
    else:
        args = [
            str(EFLOMAL_PATH),
            "-s",
            str(source_path),
            "-t",
            str(target_path),
            "-f",
            str(forward_links_path),
            "-r",
            str(reverse_links_path),
        ]
    args.extend(
        [
            "-q",
            "-m",
            "3",
            "-n",
            "3",
            "-N",
            "0.2",
            "-1",
            str(n_iterations[0]),
            "-2",
            str(n_iterations[1]),
            "-3",
            str(n_iterations[2]),
        ]
    )
    subprocess.run(args, stderr=subprocess.DEVNULL)


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
