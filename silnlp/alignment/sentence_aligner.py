from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Sequence

from machine.translation import WordAlignmentMatrix

from ..common.corpus import load_corpus, write_corpus
from .eflomal import to_word_alignment_matrix
from .utils import compute_alignment_scores


class SentenceAligner(ABC):
    """Aligns the words of each translated sentence to the words of the sentence it came from."""

    @abstractmethod
    def align(self, source: Sequence[str], translation: Sequence[str]) -> List[WordAlignmentMatrix]: ...


class ToolSentenceAligner(SentenceAligner):
    """Aligns with one of the alignment tools, which read and write files of their own."""

    def __init__(self, aligner_id: str = "eflomal") -> None:
        self._aligner_id = aligner_id

    def align(self, source: Sequence[str], translation: Sequence[str]) -> List[WordAlignmentMatrix]:
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir, "src_align.txt")
            translation_path = Path(temp_dir, "trg_align.txt")
            alignment_path = Path(temp_dir, "sym-align.txt")
            write_corpus(source_path, source)
            write_corpus(translation_path, translation)
            compute_alignment_scores(source_path, translation_path, self._aligner_id, alignment_path)

            return [to_word_alignment_matrix(line) for line in load_corpus(alignment_path)]
