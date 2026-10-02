from pathlib import Path

import pytest

from silnlp.alignment.preprocess import load_parallel_corpus
from silnlp.common.environment import SilNlpEnv


def _create_environment(tmp_path: Path, src_dir_name: str, trg_dir_name: str) -> SilNlpEnv:
    for dir_name, name, text in ((src_dir_name, "en-AAA", "source"), (trg_dir_name, "es-BBB", "objetivo")):
        path = tmp_path / "MT" / dir_name / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{text} 0\n{text} 1\n", encoding="utf-8")
    return SilNlpEnv.create_environment_with_mt_dir(tmp_path / "MT")


@pytest.mark.parametrize("src_dir_name,trg_dir_name", [("scripture", "corpora"), ("corpora", "scripture")])
def test_pairs_of_a_scripture_and_a_basic_corpus_are_rejected(tmp_path: Path, src_dir_name: str, trg_dir_name: str):
    environment = _create_environment(tmp_path, src_dir_name, trg_dir_name)

    with pytest.raises(RuntimeError, match="same type of data"):
        load_parallel_corpus({"src": "en-AAA", "trg": "es-BBB"}, environment)


@pytest.mark.parametrize("settings", [{"corpus_books": "GEN"}, {"by_book": True}])
def test_basic_corpora_reject_settings_that_select_verses(tmp_path: Path, settings: dict):
    environment = _create_environment(tmp_path, "corpora", "corpora")

    with pytest.raises(RuntimeError, match="no verse references"):
        load_parallel_corpus({"src": "en-AAA", "trg": "es-BBB", "by_book": False, **settings}, environment)


@pytest.mark.parametrize("dir_name,expected_is_scripture", [("corpora", False), ("scripture", True)])
def test_pairs_of_one_kind_load_as_that_kind(tmp_path: Path, dir_name: str, expected_is_scripture: bool):
    environment = _create_environment(tmp_path, dir_name, dir_name)

    corpus, is_scripture = load_parallel_corpus({"src": "en-AAA", "trg": "es-BBB", "by_book": False}, environment)

    assert is_scripture == expected_is_scripture
    with corpus.get_rows() as rows:
        assert [(row.source_text, row.target_text) for row in rows] == [
            ("source 0", "objetivo 0"),
            ("source 1", "objetivo 1"),
        ]
