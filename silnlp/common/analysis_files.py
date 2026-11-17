from pathlib import Path
from typing import List


class AnalysisFiles:
    """The files the analysis of an experiment's corpora reads and writes."""

    def __init__(self, exp_dir: Path) -> None:
        self._exp_dir = exp_dir

    def directory(self) -> Path:
        return self._exp_dir

    def corpus_stats(self) -> Path:
        return self._exp_dir / "corpus-stats.csv"

    def verse_counts(self) -> Path:
        return self._exp_dir / "verse_counts.csv"

    def verse_percentages(self) -> Path:
        return self._exp_dir / "verse_percentages.csv"

    def analysis(self) -> Path:
        return self._exp_dir / f"{self._exp_dir.stem}_analysis.xlsx"

    def alignment_breakdown(self) -> Path:
        return self._exp_dir / f"{self._exp_dir.stem}_alignment_breakdown.xlsx"

    def pair_scores(self, src_project: str, trg_project: str) -> Path:
        return self._exp_dir / f"{src_project}_{trg_project}.csv"

    def score_files(self) -> List[Path]:
        return list(self._exp_dir.glob("*.csv"))
