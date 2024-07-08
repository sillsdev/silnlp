import logging
import shutil
from contextlib import ExitStack
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Iterable

from machine.corpora import AlignedWordPair
from machine.translation import SymmetrizationHeuristic, WordAlignmentMatrix

from ..common.corpus import load_corpus
from ..common.packages_utils import is_eflomal_available
from .aligner import Aligner
from .lexicon import Lexicon
from .tools import execute_atools, execute_eflomal, is_atools_available

if is_eflomal_available():
    from eflomal import read_text, write_text

LOGGER = logging.getLogger(__name__)


class EflomalAligner(Aligner):
    def __init__(self, model_dir: Path) -> None:
        if not is_eflomal_available():
            raise RuntimeError("eflomal is not installed.")

        super().__init__("eflomal", model_dir)

    def train(self, src_file_path: Path, trg_file_path: Path) -> None:
        LOGGER.info("Generating alignments")
        self.model_dir.mkdir(exist_ok=True)
        with TemporaryDirectory() as temp_dir:
            src_eflomal_path = Path(temp_dir, "source")
            trg_eflomal_path = Path(temp_dir, "target")
            with ExitStack() as stack:
                src_input_file = stack.enter_context(src_file_path.open("r", encoding="utf-8-sig"))
                trg_input_file = stack.enter_context(trg_file_path.open("r", encoding="utf-8-sig"))
                src_output_file = stack.enter_context(src_eflomal_path.open("wb"))
                trg_output_file = stack.enter_context(trg_eflomal_path.open("wb"))
                # Write input files for the eflomal binary
                n_sentences = prepare_files(src_input_file, src_output_file, trg_input_file, trg_output_file)

            iters = max(2, int(round(1.0 * 5000 / sqrt(n_sentences))))
            iters4 = max(1, iters // 4)
            n_iterations = (max(2, iters4), iters4, iters)

            # Run wrapper for the eflomal binary
            execute_eflomal(
                src_eflomal_path,
                trg_eflomal_path,
                self.model_dir / "forward-align.txt",
                self.model_dir / "reverse-align.txt",
                n_iterations,
            )
        shutil.copyfile(src_file_path, self.model_dir / "src.txt")
        shutil.copyfile(trg_file_path, self.model_dir / "trg.txt")

    def align(self, out_file_path: Path, sym_heuristic: str = "grow-diag-final-and") -> None:
        LOGGER.info("Symmetrizing alignments")
        forward_align_path = self.model_dir / "forward-align.txt"
        reverse_align_path = self.model_dir / "reverse-align.txt"
        alignments_path = self.model_dir / "alignments.txt"

        if is_atools_available():
            execute_atools(forward_align_path, reverse_align_path, alignments_path, sym_heuristic)
        else:
            heuristic = SymmetrizationHeuristic[sym_heuristic.upper().replace("-", "_")]
            with ExitStack() as stack:
                forward_file = stack.enter_context(forward_align_path.open("r", encoding="utf-8-sig"))
                reverse_file = stack.enter_context(reverse_align_path.open("r", encoding="utf-8-sig"))
                out_file = stack.enter_context(alignments_path.open("w", encoding="utf-8", newline="\n"))

                for forward_line, reverse_line in zip(forward_file, reverse_file):
                    forward_matrix = to_word_alignment_matrix(forward_line.strip())
                    reverse_matrix = to_word_alignment_matrix(reverse_line.strip())
                    src_len = max(forward_matrix.row_count, reverse_matrix.row_count)
                    trg_len = max(forward_matrix.column_count, reverse_matrix.column_count)

                    forward_matrix.resize(src_len, trg_len)
                    reverse_matrix.resize(src_len, trg_len)

                    forward_matrix.symmetrize_with(reverse_matrix, heuristic)

                    out_file.write(str(forward_matrix) + "\n")

        shutil.copyfile(alignments_path, out_file_path)

    def extract_lexicon(self, out_file_path: Path) -> None:
        lexicon = self.get_direct_lexicon()
        lexicon.write(out_file_path)

    def get_direct_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        lexicon = Lexicon()
        source = load_corpus(self.model_dir / "src.txt")
        target = load_corpus(self.model_dir / "trg.txt")
        alignments = filter(lambda a: not a.startswith("#"), load_corpus(self.model_dir / "alignments.txt"))

        for src_str, trg_str, alignment_str in zip(source, target, alignments):
            src_words = src_str.split()
            trg_words = trg_str.split()
            alignment = AlignedWordPair.from_string(alignment_str)
            for src_index, trg_index in alignment:
                if src_index >= len(src_words) or trg_index >= len(trg_words):
                    continue
                src_word = src_words[src_index]
                trg_word = trg_words[trg_index]
                lexicon.increment(src_word, trg_word)
        lexicon.normalize()
        return lexicon

    def get_inverse_lexicon(self, include_special_tokens: bool = False) -> Lexicon:
        raise RuntimeError("The aligner does not have an inverse model.")


def to_word_alignment_matrix(alignment_str: str) -> WordAlignmentMatrix:
    word_pairs = AlignedWordPair.from_string(alignment_str)
    row_count = 0
    column_count = 0
    for pair in word_pairs:
        if pair.source_index + 1 > row_count:
            row_count = pair.source_index + 1
        if pair.target_index + 1 > column_count:
            column_count = pair.target_index + 1
    return WordAlignmentMatrix.from_word_pairs(row_count, column_count, word_pairs)


def to_eflomal_text_file(input: Iterable[str], output_file: IO[bytes], prefix_len: int = 0, suffix_len: int = 0) -> int:
    sents, index = read_text(input, True, prefix_len, suffix_len)
    n_sents = len(sents)
    voc_size = len(index)
    write_text(output_file, tuple(sents), voc_size)
    return n_sents


def prepare_files(
    src_input: Iterable[str], src_output_file: IO[bytes], trg_input: Iterable[str], trg_output_file: IO[bytes]
) -> int:
    n_src_sents = to_eflomal_text_file(src_input, src_output_file)
    n_trg_sents = to_eflomal_text_file(trg_input, trg_output_file)
    if n_src_sents != n_trg_sents:
        raise ValueError("Mismatched file sizes")
    return n_src_sents
