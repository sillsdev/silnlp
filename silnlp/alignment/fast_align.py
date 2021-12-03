import math
import platform
import subprocess
from pathlib import Path
from typing import List

from ..common.environment import get_env_path, wsl_path
from .aligner import Aligner
from .lexicon import Lexicon


def execute_fast_align(input_path: Path, output_path: Path, prob_table_path: Path, reverse: bool) -> None:
    fast_align_path = Path(get_env_path("FAST_ALIGN_PATH"), "fast_align")
    if not fast_align_path.is_file():
        raise RuntimeError("fast_align is not installed.")

    args: List[str]
    if platform.system() == "Windows":
        args = ["wsl", wsl_path(fast_align_path), "-i", wsl_path(input_path), "-p", wsl_path(prob_table_path)]
    else:
        args = [str(fast_align_path), "-i", str(input_path), "-p", str(prob_table_path)]
    args.extend(["-d", "-o", "-v", "-t", "-18"])
    if reverse:
        args.append("-r")

    with output_path.open("w") as output_file:
        subprocess.run(args, stdout=output_file, stderr=subprocess.DEVNULL)


def execute_atools(forward_align_path: Path, reverse_align_path: Path, output_path: Path, sym_heuristic: str) -> None:
    atools_path = Path(get_env_path("FAST_ALIGN_PATH"), "atools")
    if not atools_path.is_file():
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
        args = [str(atools_path), "-i", str(forward_align_path), "-j", str(reverse_align_path)]
    args.extend(["-c", sym_heuristic])

    with output_path.open("w") as output_file:
        subprocess.run(args, stdout=output_file, stderr=subprocess.DEVNULL)


def load_prob_table(table_path: Path, include_special_tokens: bool) -> Lexicon:
    lexicon = Lexicon()
    with table_path.open("r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            src_word, trg_word, prob_str = line.split("\t", maxsplit=3)
            if include_special_tokens or src_word != "<eps>":
                prob = math.exp(float(prob_str))
                if prob > 0.01:
                    lexicon[src_word, trg_word] = prob
    return lexicon


class FastAlign(Aligner):
    def __init__(self, model_dir: Path) -> None:
        super().__init__("clab_fast_align", model_dir)

    @property
    def forward_prob_table_path(self) -> Path:
        return self.model_dir / "forward-prob-table.txt"

    @property
    def reverse_prob_table_path(self) -> Path:
        return self.model_dir / "reverse-prob-table.txt"

    def train(self, src_file_path: Path, trg_file_path: Path) -> None:
        self.model_dir.mkdir(exist_ok=True)
        align_input_path = self.model_dir / "align-input.txt"

        with src_file_path.open("r", encoding="utf-8") as src_tok_output_file, trg_file_path.open(
            "r", encoding="utf-8"
        ) as trg_tok_output_file, align_input_path.open("w", encoding="utf-8", newline="\n") as align_input_file:
            for src_sentence, trg_sentence in zip(src_tok_output_file, trg_tok_output_file):
                align_input_file.write(f"{src_sentence.strip()} ||| {trg_sentence.strip()}\n")

        print("Training forward model...", end="", flush=True)
        forward_align_path = self.model_dir / "forward-align.txt"

        execute_fast_align(align_input_path, forward_align_path, self.forward_prob_table_path, reverse=False)
        print(" done.")

        print("Training reverse model...", end="", flush=True)
        reverse_align_path = self.model_dir / "reverse-align.txt"
        execute_fast_align(align_input_path, reverse_align_path, self.reverse_prob_table_path, reverse=True)
        print(" done.")

    def align(self, out_file_path: Path, sym_heuristic: str = "grow-diag-final-and") -> None:
        print("Generating alignments...", end="", flush=True)
        forward_align_path = self.model_dir / "forward-align.txt"
        reverse_align_path = self.model_dir / "reverse-align.txt"
        execute_atools(forward_align_path, reverse_align_path, out_file_path, sym_heuristic)
        print(" done.")

    def get_direct_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        return load_prob_table(self.forward_prob_table_path, include_special_tokens)

    def get_inverse_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        return load_prob_table(self.reverse_prob_table_path, include_special_tokens)

    def extract_lexicon(self, out_file_path: Path) -> None:
        direct_lexicon = self.get_direct_lexicon()
        inverse_lexicon = self.get_inverse_lexicon()
        print("Symmetrizing lexicons...", end="", flush=True)
        lexicon = Lexicon.symmetrize(direct_lexicon, inverse_lexicon)
        print(" done.")
        lexicon.write(out_file_path)
