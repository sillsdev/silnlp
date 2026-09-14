import logging
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable, List

import pandas as pd
from machine.tokenization import LatinWordTokenizer

from .experiment_files import ExperimentFiles

LOGGER = logging.getLogger(__name__)


@dataclass
class SideMeasurements:
    tokens_per_verse: List[int] = field(default_factory=list)
    chars_per_token: List[int] = field(default_factory=list)
    chars_per_verse: List[int] = field(default_factory=list)
    words_per_verse: List[int] = field(default_factory=list)
    chars_per_word: List[int] = field(default_factory=list)


class TokenizationStatistics:
    _LONG_VERSE_TOKENS = 200
    _SIDES = ("Source", "Target")

    def __init__(self, files: ExperimentFiles) -> None:
        self._files = files

    def write(self) -> None:
        LOGGER.info("Calculating tokenization statistics")
        source = self._measure(self._files.tokenized_source_files(), self._files.detokenized_source_files())
        target = self._measure(self._files.tokenized_target_files(), self._files.detokenized_target_files())

        report = self._existing_report()
        report = pd.concat([report, self._tokens_per_verse(source, target)], axis=1)
        for header, of_source, of_target in [
            ("Characters/Verse", source.chars_per_verse, target.chars_per_verse),
            ("Characters/Token", source.chars_per_token, target.chars_per_token),
            ("Words/Verse", source.words_per_verse, target.words_per_verse),
            ("Characters/Word", source.chars_per_word, target.chars_per_word),
        ]:
            report = pd.concat([report, self._distribution(header, of_source, of_target)], axis=1)

        report.to_csv(self._files.statistics_report(), index=False)
        report.to_excel(self._files.statistics_spreadsheet())

    def _existing_report(self) -> pd.DataFrame:
        report_path = self._files.statistics_report()
        if report_path.is_file():
            return pd.read_csv(report_path, header=[0, 1])
        return pd.DataFrame({(" ", "Translation Side"): list(self._SIDES)})

    def _measure(self, tokenized_paths: Iterable[Path], detokenized_paths: Iterable[Path]) -> SideMeasurements:
        measurements = SideMeasurements()
        for path in tokenized_paths:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    tokens = line.split()
                    measurements.tokens_per_verse.append(len(tokens))
                    measurements.chars_per_token.extend(len(token) for token in tokens)
        for path in detokenized_paths:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    # The line still carries its newline, which has always been counted as a character.
                    measurements.chars_per_verse.append(len(line))
                    words = " ".join(LatinWordTokenizer().tokenize(line)).split()
                    measurements.words_per_verse.append(len(words))
                    measurements.chars_per_word.extend(len(word) for word in words)
        return measurements

    def _tokens_per_verse(self, source: SideMeasurements, target: SideMeasurements) -> pd.DataFrame:
        header = "Tokens/Verse"
        distribution = self._distribution(header, source.tokens_per_verse, target.tokens_per_verse)
        long_verses = pd.DataFrame(
            {
                (header, f"Num Verses >= {self._LONG_VERSE_TOKENS} Tokens"): [
                    self._count_long_verses(source.tokens_per_verse),
                    self._count_long_verses(target.tokens_per_verse),
                ]
            }
        )
        return pd.concat([distribution, long_verses], axis=1)

    def _count_long_verses(self, tokens_per_verse: List[int]) -> int:
        return sum(count >= self._LONG_VERSE_TOKENS for count in tokens_per_verse)

    def _distribution(self, header: str, of_source: List[int], of_target: List[int]) -> pd.DataFrame:
        columns = pd.MultiIndex.from_product([[header], ["Min", "Max", "Median", "Mean", "Std Dev"]])
        return pd.DataFrame([self._summarize(of_source), self._summarize(of_target)], columns=columns)

    def _summarize(self, values: List[int]) -> list:
        return [min(values), max(values), median(values), self._rounded(mean(values)), self._rounded(stdev(values))]

    def _rounded(self, value: float) -> Decimal:
        return Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
