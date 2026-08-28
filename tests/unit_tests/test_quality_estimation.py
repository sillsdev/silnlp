import logging
from pathlib import Path
from typing import Dict, List

import pytest

from silnlp.nmt.quality_estimation import NO_LINREGRESS_WARNING, estimate_quality

LOW_BOOK_CONFIDENCE = 0.3
HIGH_BOOK_CONFIDENCE = 0.8


def write_tsv(path: Path, header: str, rows: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header}\tConfidence\n")
        for key, confidence in rows.items():
            f.write(f"{key}\t{confidence}\n")


def make_confidence_files(directory: Path) -> List[Path]:
    """Write the confidence files the translate step produces for two drafted books."""
    confidence_file_paths: List[Path] = []
    for book, confidence in (("MAT", HIGH_BOOK_CONFIDENCE), ("MRK", LOW_BOOK_CONFIDENCE)):
        confidence_file_path = directory / f"41{book}.SFM.confidences.tsv"
        # Quality estimation only checks that this file exists; it reads the siblings below.
        confidence_file_path.touch()
        write_tsv(
            confidence_file_path.with_suffix(".verses.tsv"),
            "VRef",
            {f"{book} 1:1": confidence, f"{book} 1:2": confidence},
        )
        write_tsv(confidence_file_path.with_suffix(".chapters.tsv"), "Chapter", {"1": confidence})
        confidence_file_paths.append(confidence_file_path)

    write_tsv(
        directory / "confidences.books.tsv",
        "Book",
        {"MAT": HIGH_BOOK_CONFIDENCE, "MRK": LOW_BOOK_CONFIDENCE},
    )
    return confidence_file_paths


def write_linregress_file(path: Path) -> None:
    path.write_text('{"version": "0.1", "slope": 50.0, "intercept": 20.0}', encoding="utf-8")


def read_rows(path: Path) -> List[List[str]]:
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]


def test_books_report_confidence_and_low_confidence_flag(tmp_path: Path) -> None:
    confidence_file_paths = make_confidence_files(tmp_path)
    linregress_path = tmp_path / "linregress.5000.json"
    write_linregress_file(linregress_path)

    estimate_quality(linregress_path, confidence_file_paths)

    header, *rows = read_rows(tmp_path / "usability_books.tsv")
    assert header == ["Book", "Confidence", "Low Confidence", "Projected chrF3", "Label"]
    assert rows == [
        ["MAT", "0.8000", "False", "60.00", "Green"],
        ["MRK", "0.3000", "True", "35.00", "Red"],
    ]

    assert read_rows(tmp_path / "usability_chapters.tsv")[0] == [
        "Book",
        "Chapter",
        "Confidence",
        "Projected chrF3",
        "Label",
    ]
    assert read_rows(tmp_path / "usability_verses.tsv")[0] == [
        "Book",
        "Chapter",
        "Verse",
        "Confidence",
        "Projected chrF3",
        "Label",
    ]


@pytest.mark.parametrize("pass_directory_without_linregress", [False, True])
def test_no_linregress_file_reports_confidence_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, pass_directory_without_linregress: bool
) -> None:
    confidence_file_paths = make_confidence_files(tmp_path)
    linregress_path = tmp_path if pass_directory_without_linregress else None

    with caplog.at_level(logging.WARNING):
        estimate_quality(linregress_path, confidence_file_paths)

    assert any(NO_LINREGRESS_WARNING in record.message for record in caplog.records)

    header, *rows = read_rows(tmp_path / "usability_books.tsv")
    assert header == ["Book", "Confidence", "Low Confidence"]
    assert rows == [["MAT", "0.8000", "False"], ["MRK", "0.3000", "True"]]

    assert read_rows(tmp_path / "usability_chapters.tsv")[0] == ["Book", "Chapter", "Confidence"]
    assert read_rows(tmp_path / "usability_verses.tsv") == [
        ["Book", "Chapter", "Verse", "Confidence"],
        ["MAT", "1", "1", "0.8000"],
        ["MAT", "1", "2", "0.8000"],
        ["MRK", "1", "1", "0.3000"],
        ["MRK", "1", "2", "0.3000"],
    ]


def test_explicitly_named_missing_linregress_file_still_raises(tmp_path: Path) -> None:
    confidence_file_paths = make_confidence_files(tmp_path)

    with pytest.raises(FileNotFoundError):
        estimate_quality(tmp_path / "linregress.5000.json", confidence_file_paths)
