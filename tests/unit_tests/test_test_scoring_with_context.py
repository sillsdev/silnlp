from pathlib import Path
from typing import List, cast

from silnlp.common.sentence_context import CONTEXT_END_TOKEN, CONTEXT_START_TOKEN
from silnlp.nmt.config import Config
from silnlp.nmt.test import load_test_data
from silnlp.nmt.tokenizer import NullTokenizer


class FakeConfig:
    """Just the parts of Config that load_test_data reads."""

    def __init__(self, exp_dir: Path, context_size: int):
        self.exp_dir = exp_dir
        self._context_size = context_size

    @property
    def use_context(self) -> bool:
        return self._context_size > 0


def windowed(before: List[str], central: str, after: List[str]) -> str:
    parts = before + [CONTEXT_START_TOKEN, central, CONTEXT_END_TOKEN] + after
    return " ".join(parts)


def write(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


PREDICTIONS = [
    windowed(["pred zero"], "pred one", ["pred two"]),
    windowed(["pred one"], "pred two", ["pred three"]),
    "pred three with no markers at all",
]
REFERENCES = [
    windowed(["ref zero"], "ref one", ["ref two"]),
    windowed(["ref one"], "ref two", ["ref three"]),
    windowed(["ref two"], "ref three", []),
]


def run_load_test_data(tmp_path: Path, context_size: int):
    write(tmp_path / "test.trg-predictions.txt.100", PREDICTIONS)
    write(tmp_path / "test.trg.detok.txt", REFERENCES)
    config = cast(Config, FakeConfig(tmp_path, context_size))
    return load_test_data(
        NullTokenizer(),
        "test.vref.txt",
        "test.trg-predictions.txt.100",
        "test.trg-predictions.txt.100.confidences.tsv",
        "test.trg.detok*.txt",
        "test.trg-predictions.detok.txt.100",
        set(),
        config,
        {},
        False,
    )


def test_only_the_central_sentence_is_scored(tmp_path: Path):
    sys, refs, _ = run_load_test_data(tmp_path, context_size=1)
    assert sys == ["pred one", "pred two", "pred three with no markers at all"]
    assert refs == [["ref one", "ref two", "ref three"]]


def test_the_written_predictions_hold_the_central_sentence(tmp_path: Path):
    run_load_test_data(tmp_path, context_size=1)
    written = (tmp_path / "test.trg-predictions.detok.txt.100").read_text(encoding="utf-8").splitlines()
    assert written == ["pred one", "pred two", "pred three with no markers at all"]


def test_windows_are_left_alone_when_context_is_off(tmp_path: Path):
    sys, refs, _ = run_load_test_data(tmp_path, context_size=0)
    assert sys == PREDICTIONS
    assert refs == [REFERENCES]
