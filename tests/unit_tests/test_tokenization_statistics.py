from pathlib import Path
from statistics import StatisticsError

import pandas as pd
import pytest

from silnlp.nmt.corpus_inventory import CorpusInventory
from silnlp.nmt.experiment_files import ExperimentFiles
from silnlp.nmt.tokenization_statistics import TokenizationStatistics


@pytest.fixture
def exp_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "exp"
    directory.mkdir()
    return directory


@pytest.fixture
def files(corpora, environment, exp_dir) -> ExperimentFiles:
    pairs = [corpora.pair([corpora.scripture_file("en", "BSB")], [corpora.scripture_file("es", "LBLA")])]
    inventory = CorpusInventory(pairs, include_glosses=False, environment=environment)
    return ExperimentFiles(exp_dir, inventory, multi_ref_eval=False)


def write(exp_dir: Path, name: str, lines) -> None:
    (exp_dir / name).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def read_stats(exp_dir: Path) -> pd.DataFrame:
    return pd.read_csv(exp_dir / "tokenization_stats.csv", header=[0, 1])


def write_a_minimal_corpus(exp_dir: Path) -> None:
    write(exp_dir, "train.src.txt", ["a bb", "ccc d"])
    write(exp_dir, "train.trg.txt", ["xx y", "z"])
    write(exp_dir, "train.src.detok.txt", ["ab cd", "ef"])
    write(exp_dir, "train.trg.detok.txt", ["gh ij", "kl"])


def test_the_report_covers_every_measured_distribution(files, exp_dir):
    write_a_minimal_corpus(exp_dir)
    TokenizationStatistics(files).write()

    stats = read_stats(exp_dir)
    top_headers = [header for header, _ in stats.columns]
    assert "Tokens/Verse" in top_headers
    assert "Characters/Verse" in top_headers
    assert "Characters/Token" in top_headers
    assert "Words/Verse" in top_headers
    assert "Characters/Word" in top_headers
    assert list(stats[(" ", "Translation Side")]) == ["Source", "Target"]


def test_tokens_per_verse_are_counted_from_the_tokenized_files(files, exp_dir):
    write_a_minimal_corpus(exp_dir)
    TokenizationStatistics(files).write()

    stats = read_stats(exp_dir)
    # Source verses have two tokens each; target verses have two and one.
    assert list(stats[("Tokens/Verse", "Min")]) == [2, 1]
    assert list(stats[("Tokens/Verse", "Max")]) == [2, 2]


def test_characters_per_verse_includes_the_line_ending(files, exp_dir):
    write(exp_dir, "train.src.txt", ["a", "b"])
    write(exp_dir, "train.trg.txt", ["c", "d"])
    write(exp_dir, "train.src.detok.txt", ["abc", "abc"])
    write(exp_dir, "train.trg.detok.txt", ["de", "de"])

    TokenizationStatistics(files).write()

    stats = read_stats(exp_dir)
    # "abc" measures 4 and "de" measures 3, because each line still carries its newline.
    assert list(stats[("Characters/Verse", "Min")]) == [4, 3]


def test_long_verses_are_counted_separately(files, exp_dir):
    write(exp_dir, "train.src.txt", [" ".join(["tok"] * 200), "short"])
    write(exp_dir, "train.trg.txt", ["short", "short"])
    write(exp_dir, "train.src.detok.txt", ["a", "b"])
    write(exp_dir, "train.trg.detok.txt", ["a", "b"])

    TokenizationStatistics(files).write()

    stats = read_stats(exp_dir)
    assert list(stats[("Tokens/Verse", "Num Verses >= 200 Tokens")]) == [1, 0]


def test_every_tokenized_file_in_the_experiment_is_measured(files, exp_dir):
    write(exp_dir, "train.src.txt", ["a", "b"])
    write(exp_dir, "test.src.txt", ["a b c"])
    write(exp_dir, "train.trg.txt", ["a", "b"])
    write(exp_dir, "train.src.detok.txt", ["a", "b"])
    write(exp_dir, "train.trg.detok.txt", ["a", "b"])

    TokenizationStatistics(files).write()

    stats = read_stats(exp_dir)
    # The three-token verse is only in test.src.txt, so reaching it means that file was read too.
    assert list(stats[("Tokens/Verse", "Max")]) == [3, 1]


def test_a_side_with_a_single_verse_cannot_be_summarized(files, exp_dir):
    # Standard deviation needs two values, so a one-verse corpus fails rather than reporting.
    write(exp_dir, "train.src.txt", ["a"])
    write(exp_dir, "train.trg.txt", ["a"])
    write(exp_dir, "train.src.detok.txt", ["a"])
    write(exp_dir, "train.trg.detok.txt", ["a"])

    with pytest.raises(StatisticsError):
        TokenizationStatistics(files).write()


def test_a_spreadsheet_is_written_alongside_the_csv(files, exp_dir):
    write_a_minimal_corpus(exp_dir)
    TokenizationStatistics(files).write()
    assert (exp_dir / "tokenization_stats.xlsx").is_file()


def test_the_report_is_added_to_one_that_already_exists(files, exp_dir):
    write_a_minimal_corpus(exp_dir)
    existing = pd.DataFrame({(" ", "Translation Side"): ["Source", "Target"], ("Vocab", "Added"): [7, 9]})
    existing.to_csv(exp_dir / "tokenization_stats.csv", index=False)

    TokenizationStatistics(files).write()

    stats = read_stats(exp_dir)
    assert list(stats[("Vocab", "Added")]) == [7, 9]
    assert "Tokens/Verse" in [header for header, _ in stats.columns]
