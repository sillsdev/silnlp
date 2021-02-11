from typing import Optional

from .aligner import Aligner
from .lexicon import Lexicon


class ClearAligner(Aligner):
    def __init__(self, model_dir: str) -> None:
        super().__init__("clear", model_dir)

    def align(self, src_file_path: str, trg_file_path: str, out_file_path: str) -> None:
        print("Not implemented")

    def extract_lexicon(self, out_file_path: str) -> None:
        print("Not implemented")
