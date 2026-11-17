from pathlib import Path

import pytest

from silnlp.nmt.prediction_files import PredictionFile


def test_a_draft_is_numbered_before_the_extension_not_after_the_step():
    predictions = Path("/exp/test.trg-predictions.txt.5000")

    assert PredictionFile(predictions).draft(1) == Path("/exp/test.trg-predictions.1.txt.5000")
    assert PredictionFile(predictions).draft(2) == Path("/exp/test.trg-predictions.2.txt.5000")


def test_the_language_pair_and_detokenization_are_left_where_they_are():
    assert PredictionFile(Path("/exp/test.en.es.trg-predictions.detok.txt.5000")).draft(3) == Path(
        "/exp/test.en.es.trg-predictions.detok.3.txt.5000"
    )


def test_a_book_filtered_step_keeps_its_books():
    assert PredictionFile(Path("/exp/test.trg-predictions.txt.MAT_MRK-5000")).draft(1) == Path(
        "/exp/test.trg-predictions.1.txt.MAT_MRK-5000"
    )


def test_an_averaged_checkpoint_is_numbered_the_same_way():
    assert PredictionFile(Path("/exp/test.trg-predictions.txt.avg")).draft(1) == Path(
        "/exp/test.trg-predictions.1.txt.avg"
    )


def test_a_file_that_does_not_name_a_checkpoint_is_refused():
    with pytest.raises(ValueError):
        PredictionFile(Path("/exp/some-draft.txt")).draft(1)
