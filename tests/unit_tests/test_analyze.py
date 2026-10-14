from pathlib import Path
from typing import List

import pandas as pd

from silnlp.common.analyze import analyze
from silnlp.common.collect_verse_counts import DT_CANON, NT_CANON, OT_CANON
from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.seq2seq_config import Seq2SeqConfig


def _write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scripture(environment: SilNlpEnv, name: str, verses: List[str]) -> None:
    # Verse counting skips any extract shorter than a whole Bible's worth of lines.
    verse_count = len((environment.assets_dir / "vref.txt").read_text(encoding="utf-8").splitlines())
    _write_lines(environment.mt_scripture_dir / f"{name}.txt", verses + [""] * (verse_count - len(verses)))


def _write_verse_count_cache(environment: SilNlpEnv) -> None:
    # Counting into a fresh cache trips a pandas dtype error, so start from an existing one, as the bucket does.
    books = OT_CANON + NT_CANON + DT_CANON
    cache_dir = environment.mt_experiments_dir / "verse_counts"
    cache_dir.mkdir(parents=True)
    counts_columns = ["Books", "Total", "OT", "NT", "DT"] + books
    pd.DataFrame([[0] * len(counts_columns)], columns=counts_columns, index=pd.Index(["other"], name="file")).to_csv(
        cache_dir / "verse_counts.csv"
    )
    percentages_columns = ["Total", "OT", "NT", "DT"] + books
    pd.DataFrame(
        [[0.0] * len(percentages_columns)], columns=percentages_columns, index=pd.Index(["other"], name="file")
    ).to_csv(cache_dir / "verse_percentages.csv")


def test_analyze_gives_each_basic_corpus_pair_its_own_stats_alongside_scripture(tmp_path: Path):
    mt_dir = tmp_path / "MT"
    environment = SilNlpEnv.create_environment_with_mt_dir(mt_dir)
    _write_verse_count_cache(environment)
    _write_scripture(environment, "en-AAA", [f"source {i}" for i in range(8)])
    _write_scripture(environment, "es-BBB", [f"objetivo {i}" for i in range(8)])
    for src_name, trg_name in (("en-CCC", "es-DDD"), ("en-EEE", "es-FFF")):
        _write_lines(mt_dir / "corpora" / f"{src_name}.txt", ["source 0", "source 1", "", "source 3", "...", ""])
        _write_lines(mt_dir / "corpora" / f"{trg_name}.txt", ["objetivo 0", "", "objetivo 2", "objetivo 3", "x", ""])
    exp_dir = environment.mt_experiments_dir / "exp"
    exp_dir.mkdir(parents=True)
    corpus_pairs = [
        {"src": "en-AAA", "trg": "es-BBB"},
        {"src": "en-CCC", "trg": "es-DDD"},
        {"src": "en-EEE", "trg": "es-FFF"},
    ]
    config = Seq2SeqConfig(exp_dir, {"data": {"corpus_pairs": corpus_pairs}}, environment)

    analyze(config, "exp", create_summaries=True, environment=environment)

    stats = pd.read_csv(exp_dir / "corpus-stats.csv", index_col=["src_project", "trg_project"])
    assert set(stats.index) == {("en-AAA", "es-BBB"), ("en-CCC", "es-DDD"), ("en-EEE", "es-FFF")}
    for basic_pair in (("en-CCC", "es-DDD"), ("en-EEE", "es-FFF")):
        assert stats.loc[basic_pair, ["count", "src_only", "trg_only", "parallel"]].tolist() == [5, 1, 2, 2]
    assert (exp_dir / "exp_alignment_breakdown.xlsx").is_file()
    assert (exp_dir / "exp_analysis.xlsx").is_file()
