from pathlib import Path
from typing import List

import pandas as pd
import pytest

from silnlp.nmt.alignment_scores import AlignmentScores, CorpusAligner


class ScriptedAligner(CorpusAligner):
    """Scores a corpus without running a real aligner, recording what it was asked to align."""

    def __init__(self, scores: List[float]) -> None:
        self._scores = scores
        self.aligned_corpora: List[int] = []

    def add_scores(self, corpus: pd.DataFrame, aligner_id: str) -> None:
        self.aligned_corpora.append(len(corpus))
        corpus["score"] = self._scores[: len(corpus)]


def corpus_of(size: int) -> pd.DataFrame:
    return pd.DataFrame({"source": [f"s{i}" for i in range(size)], "target": [f"t{i}" for i in range(size)]})


def scores_for(tmp_path: Path, aligner: CorpusAligner, force: bool = False) -> AlignmentScores:
    return AlignmentScores(tmp_path, aligner_id="fast_align", force=force, aligner=aligner)


def test_scores_are_computed_and_cached_when_none_have_been_recorded(tmp_path):
    aligner = ScriptedAligner([0.9, 0.1])
    corpus = corpus_of(2)

    scores_for(tmp_path, aligner).add_to(corpus, "en-BSB", "es-LBLA")

    assert list(corpus["score"]) == [0.9, 0.1]
    assert aligner.aligned_corpora == [2]
    assert (tmp_path / "en-BSB_es-LBLA.csv").is_file()


def test_cached_scores_are_reused_rather_than_recomputed(tmp_path):
    aligner = ScriptedAligner([0.9, 0.1])
    scores_for(tmp_path, aligner).add_to(corpus_of(2), "en-BSB", "es-LBLA")

    second_aligner = ScriptedAligner([0.0, 0.0])
    corpus = corpus_of(2)
    scores_for(tmp_path, second_aligner).add_to(corpus, "en-BSB", "es-LBLA")

    assert list(corpus["score"]) == [0.9, 0.1]
    assert second_aligner.aligned_corpora == []


def test_recomputing_is_forced_when_asked(tmp_path):
    scores_for(tmp_path, ScriptedAligner([0.9, 0.1])).add_to(corpus_of(2), "en-BSB", "es-LBLA")

    second_aligner = ScriptedAligner([0.4, 0.5])
    corpus = corpus_of(2)
    scores_for(tmp_path, second_aligner, force=True).add_to(corpus, "en-BSB", "es-LBLA")

    assert list(corpus["score"]) == [0.4, 0.5]
    assert second_aligner.aligned_corpora == [2]


def test_each_language_pair_is_cached_separately(tmp_path):
    aligner = ScriptedAligner([0.9, 0.1])
    scores = scores_for(tmp_path, aligner)
    scores.add_to(corpus_of(2), "en-BSB", "es-LBLA")
    scores.add_to(corpus_of(2), "en-BSB", "fr-LSG")

    assert aligner.aligned_corpora == [2, 2]
    assert (tmp_path / "en-BSB_es-LBLA.csv").is_file()
    assert (tmp_path / "en-BSB_fr-LSG.csv").is_file()


def test_cached_scores_are_matched_to_the_corpus_by_position(tmp_path):
    # The cache holds no verse references, so the rows are assumed to be in the same order.
    aligner = ScriptedAligner([0.9, 0.1, 0.5])
    scores_for(tmp_path, aligner).add_to(corpus_of(3), "en-BSB", "es-LBLA")

    corpus = corpus_of(3)
    corpus.index = [10, 11, 12]
    scores_for(tmp_path, ScriptedAligner([])).add_to(corpus, "en-BSB", "es-LBLA")

    assert list(corpus["score"]) == [0.9, 0.1, 0.5]


@pytest.mark.parametrize(
    "threshold,kept",
    [(0.5, ["s0", "s3"]), (0.0, ["s0", "s1", "s3"])],
)
def test_a_fractional_threshold_drops_the_verses_scoring_below_it(threshold, kept, tmp_path):
    corpus = corpus_of(4)
    corpus["score"] = [0.9, 0.2, 0.0, 0.6]

    filtered = AlignmentScores(tmp_path, "fast_align", force=False, aligner=ScriptedAligner([])).filter(
        corpus, threshold
    )

    assert list(filtered["source"]) == kept
    assert "score" not in filtered.columns


def test_a_whole_number_threshold_drops_that_many_of_the_worst_scoring_verses(tmp_path):
    corpus = corpus_of(4)
    corpus["score"] = [0.9, 0.2, 0.0, 0.6]

    filtered = AlignmentScores(tmp_path, "fast_align", force=False, aligner=ScriptedAligner([])).filter(corpus, 2)

    assert sorted(filtered["source"]) == ["s0", "s3"]


def test_a_threshold_larger_than_the_corpus_keeps_everything(tmp_path):
    corpus = corpus_of(2)
    corpus["score"] = [0.9, 0.2]

    filtered = AlignmentScores(tmp_path, "fast_align", force=False, aligner=ScriptedAligner([])).filter(corpus, 50)

    assert len(filtered) == 2
