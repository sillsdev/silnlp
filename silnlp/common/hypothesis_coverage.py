import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence

import numpy as np
from machine.tokenization import LatinWordTokenizer
from machine.translation import SymmetrizedWordAlignmentModel

from ..alignment.machine_aligner import EflomalAligner
from .corpus import write_corpus

LOGGER = logging.getLogger(__name__)


class CoverageTokenizer:
    """Lowercased word tokens, or characters for scripts written without spaces between words."""

    def __init__(self, by_character: bool) -> None:
        self._by_character = by_character
        self._words = LatinWordTokenizer()

    @classmethod
    def for_texts(cls, texts: Sequence[str]) -> "CoverageTokenizer":
        sample = [text for text in texts if text.strip()][:2000]
        with_spaces = sum(1 for text in sample if " " in text.strip())
        return cls(by_character=len(sample) > 0 and with_spaces / len(sample) < 0.5)

    def tokenize(self, text: str) -> List[str]:
        if self._by_character:
            return [char for char in text.lower() if not char.isspace()]
        return [token.lower() for token in self._words.tokenize(text)]


class HypothesisCoverage:
    """How much of a hypothesis its source accounts for, minus the same for the reference.

    Unaligned hypothesis words point to additions, loops and untranslated text. Subtracting the reference's coverage
    under the same aligner cancels a weak aligner and how differently the two languages express things.
    """

    def __init__(
        self,
        aligner: SymmetrizedWordAlignmentModel,
        work_dir: TemporaryDirectory,
        source_tokenizer: CoverageTokenizer,
        target_tokenizer: CoverageTokenizer,
    ) -> None:
        self._aligner = aligner
        self._work_dir = work_dir
        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer

    @classmethod
    def train(
        cls, sources: Sequence[str], targets: Sequence[str], min_pairs: int = 500
    ) -> Optional["HypothesisCoverage"]:
        if len(sources) < min_pairs:
            LOGGER.warning(f"Not scoring coverage: only {len(sources)} verse pairs to train an aligner on.")
            return None
        source_tokenizer, target_tokenizer = CoverageTokenizer.for_texts(sources), CoverageTokenizer.for_texts(targets)
        LOGGER.info(f"Training an eflomal aligner on {len(sources)} verse pairs for coverage")
        work_dir = TemporaryDirectory()
        src_path, trg_path = Path(work_dir.name, "src.txt"), Path(work_dir.name, "trg.txt")
        write_corpus(src_path, (" ".join(source_tokenizer.tokenize(s)) for s in sources))
        write_corpus(trg_path, (" ".join(target_tokenizer.tokenize(t)) for t in targets))
        aligner = EflomalAligner(Path(work_dir.name, "model"))
        aligner.train(src_path, trg_path)
        # och rather than silnlp's grow-diag-final-and, which links more repeated words to the source, so coverage
        # caught less repeated text when chrF-B was tested.
        return cls(aligner.create_symmetrized_model("och"), work_dir, source_tokenizer, target_tokenizer)

    def close(self) -> None:
        # The model holds its files open until closed, and Windows can't delete open files.
        self._aligner.close()
        self._work_dir.cleanup()

    def verse_scores(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> np.ndarray:
        return self._coverage(sources, hypotheses) - self._coverage(sources, references)

    def _coverage(self, sources: Sequence[str], targets: Sequence[str]) -> np.ndarray:
        pairs = [
            (self._source_tokenizer.tokenize(s), self._target_tokenizer.tokenize(t)) for s, t in zip(sources, targets)
        ]
        coverage = []
        for (_, target_tokens), word_pairs in zip(pairs, self._aligner.get_best_aligned_word_pairs_batch(pairs)):
            # A target word counts by its most probable link, so words aligned only because the aligner must align
            # something (an untranslated copy of the source, say) count for little.
            best: Dict[int, float] = {}
            for pair in word_pairs:
                if pair.source_index >= 0 and pair.target_index >= 0:
                    best[pair.target_index] = max(best.get(pair.target_index, 0.0), pair.translation_score)
            coverage.append(sum(best.values()) / len(target_tokens) if target_tokens else 0.0)
        return np.array(coverage)
