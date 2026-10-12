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
        self,
        pairs: List[dict],
        force_align: bool = False,
        data_settings: Optional[dict] = None,
        eval_settings: Optional[dict] = None,
    ) -> None:
        # Parsing a corpus pair consumes parts of its settings, and the comparisons reuse them.
        data = {"corpus_pairs": deepcopy(pairs), "tokenize": False, **(data_settings or {})}
        environment = SilNlpEnv.create_environment_with_mt_dir(self._mt_dir)
        config = Seq2SeqConfig(self._exp_dir, {"data": data, "eval": eval_settings or {}}, environment)
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


def test_corpora_and_scripture_pairs_write_the_same_train_test_and_val_sets(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(tmp_path, {"test_size": 2, "val_size": 2})

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_train_only_data_sets(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(tmp_path, {"type": "train"})

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_test_only_data_sets(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(tmp_path, {"type": "test", "test_size": 3})

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_data_sets_without_a_test_set(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(tmp_path, {"type": "train,val", "val_size": 3})

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_tagged_data_sets(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(
        tmp_path, {"test_size": 2, "val_size": 2, "tags": "tag"}
    )

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_data_sets_under_a_score_threshold_below_one(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(
        tmp_path, {"test_size": 2, "val_size": 2, "score_threshold": 0.5}
    )

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_data_sets_under_a_score_threshold_of_one_or_more(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(
        tmp_path, {"test_size": 2, "val_size": 2, "score_threshold": 2}
    )

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_data_sets_for_a_source_left_out_of_the_test_set(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(
        tmp_path, {"src": [{"name": "en-AAA", "include_test": False}], "test_size": 2, "val_size": 2}
    )

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_noised_data_sets(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(
        tmp_path, {"test_size": 2, "val_size": 2, "src_noise": [{"replacement": [1.0, "<blank>"]}]}
    )

    assert corpora_data_sets == scripture_data_sets


def test_corpora_and_scripture_pairs_write_the_same_data_sets_with_a_fractional_size(tmp_path: Path):
    corpora_data_sets, scripture_data_sets = preprocess_both_kinds(
        tmp_path, {"test_size": 2, "val_size": 2, "size": 0.5}
    )

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


def test_multi_reference_val_files_stay_aligned_when_preprocessed_again(tmp_path: Path):
    experiment = PreprocessingExperiment(tmp_path, "scripture")
    experiment.write_corpus(8)
    experiment.add_corpus("scripture", "es-EEE", 8)
    experiment.add_corpus("corpora", "en-CCC", 6)
    experiment.add_corpus("corpora", "es-DDD", 6)
    pairs = [
        {"src": "en-AAA", "trg": "es-BBB", "test_size": 2, "val_size": 2},
        {"src": "en-AAA", "trg": "es-EEE", "test_size": 2, "val_size": 2},
        {"src": "en-CCC", "trg": "es-DDD", "test_size": 2, "val_size": 2},
    ]

    for _ in range(2):
        experiment.preprocess_pairs(pairs, eval_settings={"multi_ref_eval": True})

    assert experiment.count_lines_by_split()["val"] == {6}


@pytest.fixture
def basic_experiment(tmp_path: Path) -> PreprocessingExperiment:
    return PreprocessingExperiment(tmp_path, "corpora")


@pytest.fixture
def scripture_experiment(tmp_path: Path) -> PreprocessingExperiment:
    return PreprocessingExperiment(tmp_path, "scripture")


def test_score_threshold_below_one_drops_basic_train_pairs_at_or_below_it(
    basic_experiment: PreprocessingExperiment,
):
    basic_experiment.write_corpus(8)
    basic_experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.5, 0.6, 0.95])

    basic_experiment.preprocess({"score_threshold": 0.5, "test_size": 0, "val_size": 0})

    assert sorted(basic_experiment.read_sources("train")) == basic_experiment.sources([0, 2, 4, 6, 7])


def test_score_threshold_below_one_drops_scripture_train_pairs_at_or_below_it(
    scripture_experiment: PreprocessingExperiment,
):
    scripture_experiment.write_corpus(8)
    scripture_experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.5, 0.6, 0.95])

    scripture_experiment.preprocess({"score_threshold": 0.5, "test_size": 0, "val_size": 0})

    assert sorted(scripture_experiment.read_sources("train")) == scripture_experiment.sources([0, 2, 4, 6, 7])


def test_score_threshold_of_one_or_more_drops_that_many_lowest_scoring_basic_train_pairs(
    basic_experiment: PreprocessingExperiment,
):
    basic_experiment.write_corpus(8)
    basic_experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95])

    basic_experiment.preprocess({"score_threshold": 2, "test_size": 0, "val_size": 0})

    assert sorted(basic_experiment.read_sources("train")) == basic_experiment.sources([0, 2, 4, 5, 6, 7])


def test_score_threshold_of_one_or_more_drops_that_many_lowest_scoring_scripture_train_pairs(
    scripture_experiment: PreprocessingExperiment,
):
    scripture_experiment.write_corpus(8)
    scripture_experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95])

    scripture_experiment.preprocess({"score_threshold": 2, "test_size": 0, "val_size": 0})

    assert sorted(scripture_experiment.read_sources("train")) == scripture_experiment.sources([0, 2, 4, 5, 6, 7])


def test_score_threshold_leaves_basic_test_set_unfiltered_and_draws_val_set_from_filtered_pairs(
    basic_experiment: PreprocessingExperiment,
):
    basic_experiment.write_corpus(8)
    basic_experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95])

    basic_experiment.preprocess({"score_threshold": 0.5, "test_size": 2, "val_size": 2})

    test, val, train = (basic_experiment.read_sources(split) for split in ("test", "val", "train"))
    low_scoring = set(basic_experiment.sources([1, 3, 5]))
    assert len(test) == 2 and len(val) == 2
    assert low_scoring.isdisjoint(val + train)
    assert set(basic_experiment.sources([0, 2, 4, 6, 7])) <= set(test + val + train)


def test_score_threshold_leaves_scripture_test_set_unfiltered_and_draws_val_set_from_filtered_pairs(
    scripture_experiment: PreprocessingExperiment,
):
    scripture_experiment.write_corpus(8)
    scripture_experiment.cache_scores([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.95])

    scripture_experiment.preprocess({"score_threshold": 0.5, "test_size": 2, "val_size": 2})

    test, val, train = (scripture_experiment.read_sources(split) for split in ("test", "val", "train"))
    low_scoring = set(scripture_experiment.sources([1, 3, 5]))
    assert len(test) == 2 and len(val) == 2
    assert low_scoring.isdisjoint(val + train)
    assert set(scripture_experiment.sources([0, 2, 4, 6, 7])) <= set(test + val + train)


def test_score_threshold_is_ignored_for_basic_pairs_not_used_for_training(basic_experiment: PreprocessingExperiment):
    basic_experiment.write_corpus(4)
    basic_experiment.cache_scores([0.9, 0.1, 0.1, 0.1])

    basic_experiment.preprocess({"type": "test", "score_threshold": 0.5, "test_size": 4})

    assert sorted(basic_experiment.read_sources("test")) == basic_experiment.sources(range(4))


def test_score_threshold_is_ignored_for_scripture_pairs_not_used_for_training(
    scripture_experiment: PreprocessingExperiment,
):
    scripture_experiment.write_corpus(4)
    scripture_experiment.cache_scores([0.9, 0.1, 0.1, 0.1])

    scripture_experiment.preprocess({"type": "test", "score_threshold": 0.5, "test_size": 4})

    assert sorted(scripture_experiment.read_sources("test")) == scripture_experiment.sources(range(4))


def test_basic_scores_are_computed_and_cached_when_no_cache_exists(basic_experiment: PreprocessingExperiment):
    basic_experiment.write_corpus(8)

    basic_experiment.preprocess({"score_threshold": 0.01, "test_size": 0, "val_size": 0})

    assert len(basic_experiment.read_cached_scores()) == 8


def test_scripture_scores_are_computed_and_cached_when_no_cache_exists(scripture_experiment: PreprocessingExperiment):
    scripture_experiment.write_corpus(8)

    scripture_experiment.preprocess({"score_threshold": 0.01, "test_size": 0, "val_size": 0})

    assert len(scripture_experiment.read_cached_scores()) == 8


def test_cached_basic_scores_for_a_different_corpus_are_recomputed(basic_experiment: PreprocessingExperiment):
    basic_experiment.write_corpus(8)
    basic_experiment.cache_scores([0.9] * 5)

    basic_experiment.preprocess({"score_threshold": 0.01, "test_size": 0, "val_size": 0})

    assert len(basic_experiment.read_cached_scores()) == 8


def test_cached_scripture_scores_for_a_different_corpus_are_recomputed(scripture_experiment: PreprocessingExperiment):
    scripture_experiment.write_corpus(8)
    scripture_experiment.cache_scores([0.9] * 5)

    scripture_experiment.preprocess({"score_threshold": 0.01, "test_size": 0, "val_size": 0})

    assert len(scripture_experiment.read_cached_scores()) == 8


def test_force_align_recomputes_cached_basic_scores(basic_experiment: PreprocessingExperiment):
    basic_experiment.write_corpus(8)
    basic_experiment.cache_scores([0.0] * 8)

    basic_experiment.preprocess({"score_threshold": 0.01, "test_size": 0, "val_size": 0}, force_align=True)

    assert basic_experiment.read_cached_scores() != [0.0] * 8
    assert len(basic_experiment.read_sources("train")) > 0


def test_force_align_recomputes_cached_scripture_scores(scripture_experiment: PreprocessingExperiment):
    scripture_experiment.write_corpus(8)
    scripture_experiment.cache_scores([0.0] * 8)

    scripture_experiment.preprocess({"score_threshold": 0.01, "test_size": 0, "val_size": 0}, force_align=True)

    assert scripture_experiment.read_cached_scores() != [0.0] * 8
    assert len(scripture_experiment.read_sources("train")) > 0


def test_src_noise_is_only_added_to_basic_training_sources(basic_experiment: PreprocessingExperiment):
    basic_experiment.write_corpus(8)

    basic_experiment.preprocess({"test_size": 2, "val_size": 2, "src_noise": [{"replacement": [1.0, "<blank>"]}]})

    assert set(basic_experiment.read_sources("train")) == {"<blank> <blank>"}
    held_out_sources = basic_experiment.read_sources("test") + basic_experiment.read_sources("val")
    assert set(held_out_sources) <= set(basic_experiment.sources(range(8)))


def test_src_noise_is_only_added_to_scripture_training_sources(scripture_experiment: PreprocessingExperiment):
    scripture_experiment.write_corpus(8)

    scripture_experiment.preprocess({"test_size": 2, "val_size": 2, "src_noise": [{"replacement": [1.0, "<blank>"]}]})

    assert set(scripture_experiment.read_sources("train")) == {"<blank> <blank>"}
    held_out_sources = scripture_experiment.read_sources("test") + scripture_experiment.read_sources("val")
    assert set(held_out_sources) <= set(scripture_experiment.sources(range(8)))


def test_fractional_size_is_a_fraction_of_the_basic_pairs_left_for_training(basic_experiment: PreprocessingExperiment):
    basic_experiment.write_corpus(8)

    basic_experiment.preprocess({"test_size": 2, "val_size": 2, "size": 0.5})

    assert len(basic_experiment.read_sources("train")) == 2


def test_fractional_size_is_a_fraction_of_the_scripture_pairs_left_for_training(
    scripture_experiment: PreprocessingExperiment,
):
    scripture_experiment.write_corpus(8)

    scripture_experiment.preprocess({"test_size": 2, "val_size": 2, "size": 0.5})

    assert len(scripture_experiment.read_sources("train")) == 2


def test_basic_pairs_reject_corpus_books(basic_experiment: PreprocessingExperiment):
    basic_experiment.write_corpus(8)

    with pytest.raises(RuntimeError, match="no verse references"):
        basic_experiment.preprocess({"corpus_books": "GEN"})


def test_basic_pairs_reject_test_books(basic_experiment: PreprocessingExperiment):
    basic_experiment.write_corpus(8)

    with pytest.raises(RuntimeError, match="no verse references"):
        basic_experiment.preprocess({"test_books": "GEN"})


def test_basic_pairs_reject_use_test_set_from(basic_experiment: PreprocessingExperiment):
    basic_experiment.write_corpus(8)

    with pytest.raises(RuntimeError, match="no verse references"):
        basic_experiment.preprocess({"use_test_set_from": "other"})


def test_scripture_pairs_reject_dictionary_data(scripture_experiment: PreprocessingExperiment):
    scripture_experiment.write_corpus(8)

    with pytest.raises(RuntimeError, match="dictionary or lexical"):
        scripture_experiment.preprocess({"type": "train,dict"})


def test_scripture_pairs_reject_lexical_data(scripture_experiment: PreprocessingExperiment):
    scripture_experiment.write_corpus(8)

    with pytest.raises(RuntimeError, match="dictionary or lexical"):
        scripture_experiment.preprocess({"lexical": True})
