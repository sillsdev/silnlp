from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, TextIO

from machine.scripture import VerseRef

from ...common.corpus import load_corpus


@dataclass
class VerseRange:
    start_ref: VerseRef
    end_ref: VerseRef

    def is_ref_in_range(self, ref: VerseRef) -> bool:
        return ref.compare_to(self.start_ref) >= 0 and ref.compare_to(self.end_ref) <= 0


@dataclass
class Verse:
    reference: VerseRef
    text: str


class VerseCollector(VerseRange):
    @abstractmethod
    def add_verse(self, verse: Verse) -> None: ...


@dataclass
class Passage(VerseRange):
    text: str


@dataclass
class SegmentedPassage(Passage):
    verses: List[Verse]

    def write_to_file(self, file: TextIO) -> None:
        for verse in self.verses:
            file.write(verse.text + "\n")


class SegmentedPassageBuilder(VerseCollector):
    def __init__(self, start_ref: VerseRef, end_ref: VerseRef):
        super().__init__(start_ref, end_ref)
        self._text = ""
        self._verses: List[Verse] = []

    def add_verse(self, verse: Verse) -> None:
        self._verses.append(verse)
        self._text += verse.text + " "

    def build(self) -> SegmentedPassage:
        return SegmentedPassage(self.start_ref, self.end_ref, self._text.replace("  ", " ").strip(), self._verses)


class PassageReader:
    def __init__(self, passage_file: Path):
        self._passages = self._read_passage_file(passage_file)

    def _read_passage_file(self, passage_file: Path) -> List[Passage]:
        passages = []
        for line in load_corpus(passage_file):
            row = line.split("\t")
            if len(row) < 7:
                continue  # skip malformed lines
            passage = Passage(
                VerseRef(row[0], int(row[1]), int(row[2])),
                VerseRef(row[3], int(row[4]), int(row[5])),
                text=row[6],
            )
            passages.append(passage)
        return passages

    def get_passages(self) -> List[Passage]:
        return self._passages
