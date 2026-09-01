from pathlib import Path

from silnlp.nmt.corpora import read_parallel_text_pairs


def _write_lines(path: Path, lines: list) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_parallel_text_pairs_returns_aligned_rows(tmp_path):
    src_path = tmp_path / "train.src.txt"
    trg_path = tmp_path / "train.trg.txt"
    _write_lines(src_path, ["hello", "world"])
    _write_lines(trg_path, ["bonjour", "monde"])

    pairs = read_parallel_text_pairs(src_path, trg_path)

    assert pairs == (["hello", "world"], ["bonjour", "monde"])


def test_read_parallel_text_pairs_returns_none_when_file_missing(tmp_path):
    assert read_parallel_text_pairs(tmp_path / "missing.src.txt", tmp_path / "missing.trg.txt") is None
