import shutil
from pathlib import Path

from machine.corpora import AlignedWordPair

from ..common.corpus import load_corpus
from .aligner import Aligner
from .lexicon import Lexicon


class ClearAligner(Aligner):
    def __init__(self, model_dir: Path) -> None:
        super().__init__("clear", model_dir)

    @property
    def has_inverse_model(self) -> bool:
        return False

    def train(self, src_file_path: Path, trg_file_path: Path) -> None:
        self.model_dir.mkdir(exist_ok=True)
        shutil.copyfile(src_file_path, self.model_dir / "src.txt")
        shutil.copyfile(trg_file_path, self.model_dir / "trg.txt")

    def align(self, out_file_path: Path, sym_heuristic: str = "grow-diag-final-and") -> None:
        if out_file_path.is_file():
            shutil.copyfile(out_file_path, self.model_dir / "alignments.txt")

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

    def extract_lexicon(self, out_file_path: Path) -> None:
        lexicon = self.get_direct_lexicon()
        lexicon.write(out_file_path)
