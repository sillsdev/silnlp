import logging
from pathlib import Path
from typing import List

import pytest

from silnlp.nmt.quality_estimation import BookScores, Score, compute_book_labels, get_chrf3_cells, validate_inputs


def touch_confidence_file(directory: Path, name: str = "41MAT.SFM.confidences.tsv") -> List[Path]:
    confidence_file_path = directory / name
    confidence_file_path.touch()
    return [confidence_file_path]


def test_get_chrf3_cells_when_projected_chrf3_not_included() -> None:
    assert get_chrf3_cells(60.0, False) == []


def test_get_chrf3_cells_when_projected_chrf3_is_none() -> None:
    assert get_chrf3_cells(None, True) == ["", ""]


def test_get_chrf3_cells_when_projected_chrf3_is_included() -> None:
    assert get_chrf3_cells(100.0, True) == ["100.00", "Green"]


def test_validate_inputs_with_no_linregress_path(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    confidence_file_paths = touch_confidence_file(tmp_path)

    with caplog.at_level(logging.WARNING):
        linear_regression_result, confidence_files = validate_inputs(None, confidence_file_paths)

    assert linear_regression_result is None
    assert len(confidence_files) == 1


def test_validate_inputs_with_directory_missing_linregress_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    confidence_file_paths = touch_confidence_file(tmp_path)

    with caplog.at_level(logging.WARNING):
        linear_regression_result, confidence_files = validate_inputs(tmp_path, confidence_file_paths)

    assert linear_regression_result is None
    assert len(confidence_files) == 1


def test_validate_inputs_with_explicit_missing_linregress_path(tmp_path: Path) -> None:
    confidence_file_paths = touch_confidence_file(tmp_path)

    with pytest.raises(FileNotFoundError):
        validate_inputs(tmp_path / "linregress.5000.json", confidence_file_paths)


def test_compute_book_labels_with_projected_chrf3_not_included(tmp_path: Path) -> None:
    book_scores = BookScores()
    book_scores.add_score("MAT", Score(confidence=0.8, projected_chrf3=None))
    book_scores.add_score("COL", Score(confidence=0.3, projected_chrf3=None))

    compute_book_labels(book_scores, tmp_path, include_projected_chrf3=False)

    rows = [line.split("\t") for line in (tmp_path / "usability_books.tsv").read_text(encoding="utf-8").splitlines()]
    assert rows[0] == ["Book", "Confidence", "Low Confidence"]
    assert rows[1:] == [["MAT", "0.8000", "False"], ["COL", "0.3000", "True"]]


def test_compute_book_labels_with_projected_chrf3_included(tmp_path: Path) -> None:
    book_scores = BookScores()
    book_scores.add_score("MAT", Score(confidence=0.8, projected_chrf3=60.0))
    book_scores.add_score("COL", Score(confidence=0.3, projected_chrf3=30.0))

    compute_book_labels(book_scores, tmp_path, include_projected_chrf3=True)

    rows = [line.split("\t") for line in (tmp_path / "usability_books.tsv").read_text(encoding="utf-8").splitlines()]
    assert rows[0] == ["Book", "Confidence", "Low Confidence", "Projected chrF3", "Label"]
    assert rows[1:] == [
        ["MAT", "0.8000", "False", "60.00", "Green"],
        ["COL", "0.3000", "True", "30.00", "Red"],
    ]
