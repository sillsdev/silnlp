from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pytest

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.seq2seq_config import Seq2SeqConfig


class PreprocessingExperiment:
    def __init__(self, tmp_path: Path, corpus_dir_name: str) -> None:
        self._mt_dir = tmp_path / "MT"
        self._corpus_dir = self._mt_dir / corpus_dir_name
        self._exp_dir = self._mt_dir / "experiments" / "exp"
        self._corpus_dir.mkdir(parents=True)
        self._exp_dir.mkdir(parents=True)

    def write_corpus(self, size: int) -> None:
        # A scripture file this short holds the first verses of Genesis, so both kinds line up row for row.
        self._write_lines(self._corpus_dir / "en-AAA.txt", self.sources(range(size)))
        self._write_lines(self._corpus_dir / "es-BBB.txt", [f"objetivo {i}" for i in range(size)])

    def cache_scores(self, scores: List[float]) -> None:
        pd.DataFrame({"score": scores}).to_csv(self._cache_path(), index=False)

    def read_cached_scores(self) -> List[float]:
        return list(pd.read_csv(self._cache_path())["score"])

    def preprocess(self, pair_settings: dict, force_align: bool = False) -> None:
        pair = {"src": "en-AAA", "trg": "es-BBB", **pair_settings}
        environment = SilNlpEnv.create_environment_with_mt_dir(self._mt_dir)
        config = Seq2SeqConfig(self._exp_dir, {"data": {"corpus_pairs": [pair], "tokenize": False}}, environment)
        config.set_seed()
        config.preprocess(stats=False, force_align=force_align)

    def read_sources(self, split: str) -> List[str]:
        path = self._exp_dir / f"{split}.src.txt"
        if not path.is_file():
            return []
        return path.read_text(encoding="utf-8").splitlines()

    def sources(self, indices: Iterable[int]) -> List[str]:
        return [f"source {i}" for i in indices]

    def _cache_path(self) -> Path:
        return self._exp_dir / "en-AAA_es-BBB.csv"

    def _write_lines(self, path: Path, lines: List[str]) -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(params=["corpora", "scripture"])
def experiment(request, tmp_path: Path) -> PreprocessingExperiment:
    return PreprocessingExperiment(tmp_path, request.param)


def test_score_threshold_below_one_drops_train_pairs_at_or_below_it(experiment: PreprocessingExperiment):
    experiment.write_corpus(8)
    experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.5, 0.6, 0.95])

    experiment.preprocess({"score_threshold": 0.5, "test_size": 0, "val_size": 0})

    assert sorted(experiment.read_sources("train")) == experiment.sources([0, 2, 4, 6, 7])


def test_score_threshold_of_one_or_more_drops_that_many_lowest_scoring_train_pairs(
    experiment: PreprocessingExperiment,
):
    experiment.write_corpus(8)
    experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95])

    experiment.preprocess({"score_threshold": 2, "test_size": 0, "val_size": 0})

    assert sorted(experiment.read_sources("train")) == experiment.sources([0, 2, 4, 5, 6, 7])


def test_score_threshold_leaves_test_set_unfiltered_and_draws_val_set_from_filtered_pairs(
    experiment: PreprocessingExperiment,
):
    experiment.write_corpus(8)
    experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95])

    experiment.preprocess({"score_threshold": 0.5, "test_size": 2, "val_size": 2})

    test, val, train = (experiment.read_sources(split) for split in ("test", "val", "train"))
    low_scoring = set(experiment.sources([1, 3, 5]))
    assert len(test) == 2 and len(val) == 2
    assert low_scoring.isdisjoint(val + train)
    assert set(experiment.sources([0, 2, 4, 6, 7])) <= set(test + val + train)


def test_score_threshold_is_ignored_for_pairs_not_used_for_training(experiment: PreprocessingExperiment):
    experiment.write_corpus(4)
    experiment.cache_scores([0.9, 0.1, 0.1, 0.1])

    experiment.preprocess({"type": "test", "score_threshold": 0.5, "test_size": 4})

    assert sorted(experiment.read_sources("test")) == experiment.sources(range(4))


def test_scores_are_computed_and_cached_when_no_cache_exists(experiment: PreprocessingExperiment):
    experiment.write_corpus(8)

    experiment.preprocess({"score_threshold": 0.01, "test_size": 0, "val_size": 0})

    assert len(experiment.read_cached_scores()) == 8


def test_force_align_recomputes_cached_scores(experiment: PreprocessingExperiment):
    experiment.write_corpus(8)
    experiment.cache_scores([0.0] * 8)

    experiment.preprocess({"score_threshold": 0.01, "test_size": 0, "val_size": 0}, force_align=True)

    assert experiment.read_cached_scores() != [0.0] * 8
    assert len(experiment.read_sources("train")) > 0
