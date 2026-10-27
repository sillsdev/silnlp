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


def test_a_scripture_source_paired_with_a_basic_corpus_target_is_rejected(tmp_path: Path):
    environment = _create_environment(tmp_path, "scripture", "corpora")

    with pytest.raises(RuntimeError, match="same type of data"):
        load_parallel_corpus({"src": "en-AAA", "trg": "es-BBB"}, environment)


def test_a_basic_corpus_source_paired_with_a_scripture_target_is_rejected(tmp_path: Path):
    environment = _create_environment(tmp_path, "corpora", "scripture")

    with pytest.raises(RuntimeError, match="same type of data"):
        load_parallel_corpus({"src": "en-AAA", "trg": "es-BBB"}, environment)


def test_basic_corpora_reject_corpus_books(tmp_path: Path):
    environment = _create_environment(tmp_path, "corpora", "corpora")

    with pytest.raises(RuntimeError, match="no verse references"):
        load_parallel_corpus({"src": "en-AAA", "trg": "es-BBB", "by_book": False, "corpus_books": "GEN"}, environment)


def test_basic_corpora_reject_by_book(tmp_path: Path):
    environment = _create_environment(tmp_path, "corpora", "corpora")

    with pytest.raises(RuntimeError, match="no verse references"):
        load_parallel_corpus({"src": "en-AAA", "trg": "es-BBB", "by_book": True}, environment)


def test_a_pair_of_basic_corpora_loads_as_line_pairs(tmp_path: Path):
    environment = _create_environment(tmp_path, "corpora", "corpora")

    corpus, is_scripture = load_parallel_corpus({"src": "en-AAA", "trg": "es-BBB", "by_book": False}, environment)

    assert not is_scripture
    with corpus.get_rows() as rows:
        assert [(row.source_text, row.target_text) for row in rows] == [
            ("source 0", "objetivo 0"),
            ("source 1", "objetivo 1"),
        ]


def test_a_pair_of_scripture_files_loads_as_scripture(tmp_path: Path):
    environment = _create_environment(tmp_path, "scripture", "scripture")

    corpus, is_scripture = load_parallel_corpus({"src": "en-AAA", "trg": "es-BBB", "by_book": False}, environment)

    assert is_scripture
    with corpus.get_rows() as rows:
        assert [(row.source_text, row.target_text) for row in rows] == [
            ("source 0", "objetivo 0"),
            ("source 1", "objetivo 1"),
        ]
