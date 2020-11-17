import math
import os
import platform
import subprocess
from typing import Dict, List, Tuple

from nlp.alignment.aligner import Aligner
from nlp.common.utils import wsl_path


def execute_fast_align(input_path: str, output_path: str, prob_table_path: str, reverse: bool) -> None:
    fast_align_path = os.path.join(os.getenv("FAST_ALIGN_PATH", "."), "fast_align")
    if not os.path.isfile(fast_align_path):
        raise RuntimeError("fast_align is not installed.")

    args: List[str]
    if platform.system() == "Windows":
        args = ["wsl", wsl_path(fast_align_path), "-i", wsl_path(input_path), "-p", wsl_path(prob_table_path)]
    else:
        args = [fast_align_path, "-i", input_path, "-p", prob_table_path]
    args.extend(["-d", "-o", "-v"])
    if reverse:
        args.append("-r")

    with open(output_path, "w") as output_file:
        subprocess.run(args, stdout=output_file, stderr=subprocess.DEVNULL)


def execute_atools(forward_align_path: str, reverse_align_path: str, output_path: str) -> None:
    atools_path = os.path.join(os.getenv("FAST_ALIGN_PATH", "."), "atools")
    if not os.path.isfile(atools_path):
        raise RuntimeError("atools is not installed.")

    args: List[str]
    if platform.system() == "Windows":
        args = [
            "wsl",
            wsl_path(atools_path),
            "-i",
            wsl_path(forward_align_path),
            "-j",
            wsl_path(reverse_align_path),
        ]
    else:
        args = [atools_path, "-i", forward_align_path, "-j", reverse_align_path]
    args.extend(["-c", "grow-diag-final-and"])

    with open(output_path, "w") as output_file:
        subprocess.run(args, stdout=output_file, stderr=subprocess.DEVNULL)


def load_prob_table(table_path: str) -> Dict[Tuple[str, str], float]:
    table: Dict[Tuple[str, str], float] = {}
    with open(table_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            row = line.split("\t")
            table[(row[0], row[1])] = math.exp(float(row[2]))
    return table


class FastAlign(Aligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("fast_align", model_dir)

    @property
    def forward_prob_table_path(self) -> str:
        return os.path.join(self.model_dir, "forward-prob-table.txt")

    @property
    def reverse_prob_table_path(self) -> str:
        return os.path.join(self.model_dir, "reverse-prob-table.txt")

    def align(self, src_file_path: str, trg_file_path: str, out_file_path: str) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        align_input_path = os.path.join(self.model_dir, "align-input.txt")

        with open(src_file_path, "r", encoding="utf-8") as src_tok_output_file, open(
            trg_file_path, "r", encoding="utf-8"
        ) as trg_tok_output_file, open(align_input_path, "w", encoding="utf-8", newline="\n") as align_input_file:
            for src_sentence, trg_sentence in zip(src_tok_output_file, trg_tok_output_file):
                align_input_file.write(f"{src_sentence.strip()} ||| {trg_sentence.strip()}\n")

        print("Training forward model...", end="")
        forward_align_path = os.path.join(self.model_dir, "forward-align.txt")
        execute_fast_align(align_input_path, forward_align_path, self.forward_prob_table_path, reverse=False)
        print(" done.")

        print("Training reverse model...", end="")
        reverse_align_path = os.path.join(self.model_dir, "reverse-align.txt")
        execute_fast_align(align_input_path, reverse_align_path, self.reverse_prob_table_path, reverse=True)
        print(" done.")

        print("Generating alignments...", end="")
        execute_atools(forward_align_path, reverse_align_path, out_file_path)
        print(" done.")

    def get_forward_prob_table(self) -> Dict[Tuple[str, str], float]:
        return load_prob_table(self.forward_prob_table_path)

    def get_reverse_prob_table(self) -> Dict[Tuple[str, str], float]:
        return load_prob_table(self.reverse_prob_table_path)
