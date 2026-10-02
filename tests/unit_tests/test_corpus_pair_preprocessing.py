from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
        self.write_sentence_pairs(self.sources(range(size)), [f"objetivo {i}" for i in range(size)])

    def add_corpus(self, corpus_dir_name: str, name: str, size: int) -> None:
        corpus_dir = self._mt_dir / corpus_dir_name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        self._write_lines(corpus_dir / f"{name}.txt", [f"{name} {i}" for i in range(size)])

    def write_sentence_pairs(self, sources: List[str], targets: List[str]) -> None:
        # A scripture file this short holds the first verses of Genesis, so both kinds line up row for row.
        self._write_lines(self._corpus_dir / "en-AAA.txt", sources)
        self._write_lines(self._corpus_dir / "es-BBB.txt", targets)

    def cache_scores(self, scores: List[float]) -> None:
        pd.DataFrame({"score": scores}).to_csv(self._cache_path(), index=False)

    def read_cached_scores(self) -> List[float]:
        return list(pd.read_csv(self._cache_path())["score"])

    def preprocess(self, pair_settings: dict, force_align: bool = False, data_settings: Optional[dict] = None) -> None:
        self.preprocess_pairs([{"src": "en-AAA", "trg": "es-BBB", **pair_settings}], force_align, data_settings)

    def preprocess_pairs(
        self, pairs: List[dict], force_align: bool = False, data_settings: Optional[dict] = None
    ) -> None:
        # Parsing a corpus pair consumes parts of its settings, and the comparisons reuse them.
        data = {"corpus_pairs": deepcopy(pairs), "tokenize": False, **(data_settings or {})}
        environment = SilNlpEnv.create_environment_with_mt_dir(self._mt_dir)
        config = Seq2SeqConfig(self._exp_dir, {"data": data}, environment)
        config.set_seed()
        config.preprocess(stats=False, force_align=force_align)

    def read_sources(self, split: str) -> List[str]:
        path = self._exp_dir / f"{split}.src.txt"
        if not path.is_file():
            return []
        return path.read_text(encoding="utf-8").splitlines()

    def read_data_sets(self) -> Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]:
        # Verse references are what tells the kinds apart, and training order is not meaningful.
        paths_by_split: Dict[str, List[Path]] = {}
        for path in sorted(self._exp_dir.glob("*.txt*")):
            if ".vref." not in path.name:
                paths_by_split.setdefault(path.name.split(".")[0], []).append(path)
        return {
            split: ([path.name for path in paths], sorted(zip(*(self._read_lines(path) for path in paths))))
            for split, paths in paths_by_split.items()
        }

    def count_lines_by_split(self) -> Dict[str, Set[int]]:
        line_counts: Dict[str, Set[int]] = {}
        for path in self._exp_dir.glob("*.txt*"):
            line_counts.setdefault(path.name.split(".")[0], set()).add(len(self._read_lines(path)))
        return line_counts

    def sources(self, indices: Iterable[int]) -> List[str]:
        return [f"source {i}" for i in indices]

    def _cache_path(self) -> Path:
        return self._exp_dir / "en-AAA_es-BBB.csv"

    def _write_lines(self, path: Path, lines: List[str]) -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _read_lines(self, path: Path) -> List[str]:
        return path.read_text(encoding="utf-8").splitlines()


def preprocess_both_kinds(
    tmp_path: Path, pair_settings: dict, data_settings: Optional[dict] = None
) -> List[Dict[str, Tuple[List[str], List[Tuple[str, ...]]]]]:
    data_sets = []
    for corpus_dir_name in ("corpora", "scripture"):
        experiment = PreprocessingExperiment(tmp_path / corpus_dir_name, corpus_dir_name)
        experiment.write_corpus(8)
        experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95])
        experiment.preprocess(pair_settings, data_settings=data_settings)
        data_sets.append(experiment.read_data_sets())
    return data_sets


@pytest.mark.parametrize(
    "pair_settings",
    [
        {"test_size": 2, "val_size": 2},
        {"type": "train"},
        {"type": "test", "test_size": 3},
        {"type": "train,val", "val_size": 3},
        {"test_size": 2, "val_size": 2, "tags": "tag"},
        {"test_size": 2, "val_size": 2, "score_threshold": 0.5},
        {"test_size": 2, "val_size": 2, "score_threshold": 2},
        {"src": [{"name": "en-AAA", "include_test": False}], "test_size": 2, "val_size": 2},
        {"test_size": 2, "val_size": 2, "src_noise": [{"replacement": [1.0, "<blank>"]}]},
        {"test_size": 2, "val_size": 2, "size": 0.5},
    ],
)
def test_corpora_and_scripture_pairs_write_the_same_data_sets(tmp_path: Path, pair_settings: dict):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(tmp_path, pair_settings)

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_mirrored_data_sets(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(
        tmp_path, {"test_size": 2, "val_size": 2}, {"mirror": True}
    )

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_mirrored_noised_data_sets(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(
        tmp_path, {"test_size": 2, "val_size": 2, "src_noise": [{"replacement": [1.0, "<blank>"]}]}, {"mirror": True}
    )

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_skip_the_same_empty_and_placeholder_lines(tmp_path: Path):
    train_sources = []
    for corpus_dir_name in ("corpora", "scripture"):
        experiment = PreprocessingExperiment(tmp_path / corpus_dir_name, corpus_dir_name)
        experiment.write_sentence_pairs(
            ["source 0", "...", "source 2", "", "source 4", "source 5"],
            ["objetivo 0", "objetivo 1", "...", "objetivo 3", "", "objetivo 5"],
        )
        experiment.preprocess({"type": "train"})
        train_sources.append(experiment.read_sources("train"))

    assert train_sources == [["source 0", "source 5"], ["source 0", "source 5"]]


def test_mixed_experiment_keeps_the_files_of_each_split_aligned(tmp_path: Path):
    experiment = PreprocessingExperiment(tmp_path, "scripture")
    experiment.write_corpus(8)
    experiment.add_corpus("scripture", "es-EEE", 8)
    experiment.add_corpus("corpora", "en-CCC", 6)
    experiment.add_corpus("corpora", "es-DDD", 6)

    experiment.preprocess_pairs(
        [
            {"src": "en-AAA", "trg": "es-BBB", "test_size": 2, "val_size": 2},
            {"src": "en-AAA", "trg": "es-EEE", "test_size": 2, "val_size": 2},
            {"src": "en-CCC", "trg": "es-DDD", "type": "train,test,val,dict", "test_size": 2, "val_size": 2},
        ]
    )

    assert experiment.count_lines_by_split() == {"train": {12}, "val": {6}, "test": {6}, "dict": {6}}


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


def test_src_noise_is_only_added_to_training_sources(experiment: PreprocessingExperiment):
    experiment.write_corpus(8)

    experiment.preprocess({"test_size": 2, "val_size": 2, "src_noise": [{"replacement": [1.0, "<blank>"]}]})

    assert set(experiment.read_sources("train")) == {"<blank> <blank>"}
    assert set(experiment.read_sources("test") + experiment.read_sources("val")) <= set(experiment.sources(range(8)))


def test_fractional_size_is_a_fraction_of_the_pairs_left_for_training(experiment: PreprocessingExperiment):
    experiment.write_corpus(8)

    experiment.preprocess({"test_size": 2, "val_size": 2, "size": 0.5})

    assert len(experiment.read_sources("train")) == 2


@pytest.mark.parametrize(
    "pair_settings", [{"corpus_books": "GEN"}, {"test_books": "GEN"}, {"use_test_set_from": "other"}]
)
def test_basic_pairs_reject_settings_that_select_verses(tmp_path: Path, pair_settings: dict):
    experiment = PreprocessingExperiment(tmp_path, "corpora")
    experiment.write_corpus(8)

    with pytest.raises(RuntimeError, match="no verse references"):
        experiment.preprocess(pair_settings)


@pytest.mark.parametrize("pair_settings", [{"type": "train,dict"}, {"lexical": True}])
def test_scripture_pairs_reject_dictionary_and_lexical_data(tmp_path: Path, pair_settings: dict):
    experiment = PreprocessingExperiment(tmp_path, "scripture")
    experiment.write_corpus(8)

    with pytest.raises(RuntimeError, match="dictionary or lexical"):
        experiment.preprocess(pair_settings)
