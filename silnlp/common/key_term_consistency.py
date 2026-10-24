import logging
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from machine.scripture import VerseRef
from machine.tokenization import LatinWordTokenizer

from ..alignment.machine_aligner import EflomalAligner
from .corpus import Term, write_corpus

LOGGER = logging.getLogger(__name__)


class GlossMatcher:
    """Finds a gloss in tokenized English, allowing inflectional endings ("apostle" matches "apostles")."""

    def positions(self, tokens: List[str], glosses: List[str]) -> List[int]:
        for gloss in glosses:
            gloss_tokens = gloss.lower().split()
            for start in range(len(tokens) - len(gloss_tokens) + 1):
                if all(self._matches(tokens[start + i], g) for i, g in enumerate(gloss_tokens)):
                    return list(range(start, start + len(gloss_tokens)))
        return []

    def _matches(self, token: str, gloss_token: str) -> bool:
        if token == gloss_token:
            return True
        return len(gloss_token) >= 3 and token.startswith(gloss_token) and len(token) - len(gloss_token) <= 3


class AlignedVerses:
    """English and target verses, tokenized and word-aligned, for finding how the target renders each key term."""

    def __init__(
        self,
        vrefs: List[VerseRef],
        english: List[List[str]],
        target: List[List[str]],
        alignments: List[List[Tuple[int, int]]],
    ) -> None:
        # A verse range (verses one translation combines) is found from any of its verses.
        self._row_by_verse = {str(verse): row for row, vref in enumerate(vrefs) for verse in vref.all_verses()}
        self._english = english
        self._target = target
        self._alignments = alignments
        self._matcher = GlossMatcher()
        self._rows_by_token: Dict[str, Set[int]] = {}
        for row, tokens in enumerate(target):
            for token in set(tokens):
                self._rows_by_token.setdefault(token, set()).add(row)

    @classmethod
    def align(cls, vrefs: List[VerseRef], english: Sequence[str], target: Sequence[str]) -> "AlignedVerses":
        tokenizer = LatinWordTokenizer()
        english_tokens = [[token.lower() for token in tokenizer.tokenize(verse)] for verse in english]
        target_tokens = [[token.lower() for token in tokenizer.tokenize(verse)] for verse in target]
        with TemporaryDirectory() as work_dir:
            src_path, trg_path = Path(work_dir, "src.txt"), Path(work_dir, "trg.txt")
            write_corpus(src_path, (" ".join(tokens) for tokens in english_tokens))
            write_corpus(trg_path, (" ".join(tokens) for tokens in target_tokens))
            aligner = EflomalAligner(Path(work_dir, "model"))
            aligner.train(src_path, trg_path)
            with aligner.create_symmetrized_model() as model:
                alignments = [
                    sorted((pair.source_index, pair.target_index) for pair in matrix.to_aligned_word_pairs())
                    for matrix in model.align_batch(list(zip(english_tokens, target_tokens)))
                ]
        return cls(vrefs, english_tokens, target_tokens, alignments)

    def rendering_at(self, row: int, glosses: List[str], max_span: int = 4) -> Optional[str]:
        """The target words aligned to the gloss in the row: None if the row lacks the gloss, "" if the gloss is
        unaligned or aligned to a span too long to say anything about the rendering."""
        wanted = set(self._matcher.positions(self._english[row], glosses))
        if not wanted:
            return None
        target_indices = sorted({j for i, j in self._alignments[row] if i in wanted})
        if not target_indices or target_indices[-1] - target_indices[0] >= max_span:
            return ""
        return " ".join(self._target[row][target_indices[0] : target_indices[-1] + 1])

    def rows_of(self, vrefs: Iterable[VerseRef]) -> Set[int]:
        rows = (self._row_by_verse.get(str(vref)) for vref in vrefs)
        return {row for row in rows if row is not None}

    def rows_containing(self, rendering: str) -> Set[int]:
        token_rows = [self._rows_by_token.get(token, set()) for token in rendering.split()]
        return set.intersection(*token_rows) if token_rows else set()


def infer_renderings(
    terms: Iterable[Term], verses: AlignedVerses, min_confidence: float = 0.6, min_share: float = 0.2
) -> Set[str]:
    """Each term's most frequently aligned rendering, from its English glosses, kept if it was aligned in at least
    min_confidence of the term's verses and is specific to the term.

    Alignment sometimes attaches a gloss to a frequent neighbour, such as a possessive, with high confidence. Such a
    word occurs mostly outside the term's verses, so a rendering is kept only if at least min_share of the verses
    containing it are verses of a term it renders. (Comparing raw frequencies drops frequent real renderings such as
    "God".)
    """
    inferred: List[Tuple[Set[int], str, float]] = []
    for term in terms:
        glosses = [gloss.strip() for gloss in term.glosses if gloss.strip()]
        if not glosses:
            continue
        # Each row counts once, even when it is a range holding several of the term's verses, and in verse order, so
        # that ties between equally frequent renderings break the same way every time.
        rows = verses.rows_of(term.vrefs)
        found = [verses.rendering_at(row, glosses) for row in sorted(rows)]
        found = [rendering for rendering in found if rendering is not None]
        counts = Counter(rendering for rendering in found if rendering)
        if counts:
            rendering, count = counts.most_common(1)[0]
            inferred.append((rows, rendering, count / len(found)))
    # Namesakes are separate terms (John the Baptist, John the apostle) usually sharing one rendering.
    term_rows: Dict[str, Set[int]] = {}
    for rows, rendering, _ in inferred:
        term_rows.setdefault(rendering, set()).update(rows)
    kept = set()
    for _, rendering, confidence in inferred:
        rendering_rows = verses.rows_containing(rendering)
        specific = (
            len(rendering_rows) > 0 and len(rendering_rows & term_rows[rendering]) / len(rendering_rows) >= min_share
        )
        if confidence >= min_confidence and specific:
            kept.add(rendering)
    return kept


class RenderingMatcher:
    """Finds renderings in tokenized text, tolerating suffixes on each token (inflected forms)."""

    def __init__(self, renderings: Set[str]) -> None:
        self._by_prefix: Dict[str, List[Tuple[str, ...]]] = {}
        for rendering in renderings:
            tokens = tuple(rendering.split())
            self._by_prefix.setdefault(tokens[0][:3], []).append(tokens)

    def found(self, tokens: List[str]) -> Set[Tuple[str, ...]]:
        found = set()
        for start, token in enumerate(tokens):
            for rendering in self._by_prefix.get(token[:3], []):
                if self._matches_at(tokens, start, rendering):
                    found.add(rendering)
        return found

    def contains(self, tokens: List[str], rendering: Tuple[str, ...]) -> bool:
        return any(self._matches_at(tokens, start, rendering) for start in range(len(tokens) - len(rendering) + 1))

    def _matches_at(self, tokens: List[str], start: int, rendering: Tuple[str, ...]) -> bool:
        window = tokens[start : start + len(rendering)]
        return len(window) == len(rendering) and all(self._matches(t, r) for t, r in zip(window, rendering))

    def _matches(self, token: str, rendering_token: str) -> bool:
        # Short renderings must match whole: their letters are often an inflectional prefix shared by many words.
        length = len(rendering_token)
        stem = rendering_token if length <= 5 else rendering_token[:-2]
        extra = 0 if length <= 3 else 2 if length <= 5 else 3
        return token.startswith(stem) and len(token) - length <= extra


class KeyTermConsistency:
    """The share of the key-term renderings in a reference that the hypothesis also uses."""

    def __init__(self, renderings: Set[str]) -> None:
        self._matcher = RenderingMatcher(renderings)
        self._tokenizer = LatinWordTokenizer()

    @classmethod
    def infer(
        cls,
        vrefs: List[VerseRef],
        english: Sequence[str],
        target: Sequence[str],
        terms: Iterable[Term],
        min_verses: int = 1000,
    ) -> Optional["KeyTermConsistency"]:
        if len(vrefs) < min_verses:
            LOGGER.warning(f"Not scoring key terms: only {len(vrefs)} verses to infer renderings from.")
            return None
        LOGGER.info(f"Inferring key-term renderings from {len(vrefs)} verses")
        return cls(infer_renderings(terms, AlignedVerses.align(vrefs, english, target)))

    def verse_scores(self, hypotheses: Sequence[str], references: Sequence[str]) -> np.ndarray:
        """NaN for a verse whose reference has no key term."""
        scores = []
        for hypothesis, reference in zip(hypotheses, references):
            expected = self._matcher.found(self._tokenize(reference))
            hypothesis_tokens = self._tokenize(hypothesis)
            found = [self._matcher.contains(hypothesis_tokens, rendering) for rendering in expected]
            scores.append(np.mean(found) if found else np.nan)
        return np.array(scores, dtype=float)

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in self._tokenizer.tokenize(text)]
