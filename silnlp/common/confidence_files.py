import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Generator, Generic, List, Optional, Tuple, TypeVar

from machine.corpora import ScriptureRef
from machine.scripture import VerseRef
from scipy.stats import gmean

from .translation_data_structures import TranslatedDraft

LOGGER = logging.getLogger((__package__ or "") + ".translate")

CONFIDENCE_SUFFIX = ".confidences.tsv"

TVerseKey = TypeVar("TVerseKey")


class ConfidenceFile(ABC, Generic[TVerseKey]):

    def __init__(self, path: Path):
        if not path.name.endswith(CONFIDENCE_SUFFIX):
            raise ValueError(f"Confidence file path must end with {CONFIDENCE_SUFFIX}, got {path.name}")
        self._path = path
        self._trg_draft_file_path = self._get_trg_draft_file_path_from_confidence_path(path)

    @classmethod
    def _get_confidence_file_type(cls, trg_draft_file_path: Path) -> type["ConfidenceFile"]:
        if "trg-predictions" in trg_draft_file_path.name:
            return TestConfidenceFile
        ext = trg_draft_file_path.suffix.lower()
        if ext in {".usfm", ".sfm"}:
            return UsfmConfidenceFile
        if ext == ".txt":
            return TxtConfidenceFile
        raise ValueError(
            f"No confidence file type corresponds to trg_draft_file_path {trg_draft_file_path}. "
            f"Expected a trg_draft_file_path containing 'trg-predictions' or ending with .usfm/.sfm/.txt."
        )

    @classmethod
    def from_confidence_file_path(cls, confidence_file_path: Path) -> "ConfidenceFile":
        trg_draft_file_path = cls._get_trg_draft_file_path_from_confidence_path(confidence_file_path)
        file_type = cls._get_confidence_file_type(trg_draft_file_path)
        return file_type(confidence_file_path)

    @classmethod
    def from_draft_file_path(cls, trg_draft_file_path: Path) -> "ConfidenceFile":
        confidence_file_path = trg_draft_file_path.with_suffix(f"{trg_draft_file_path.suffix}{CONFIDENCE_SUFFIX}")
        file_type = cls._get_confidence_file_type(trg_draft_file_path)
        return file_type(confidence_file_path)

    def get_path(self) -> Path:
        return self._path

    def get_verses_path(self) -> Path:
        return self._path.with_suffix(".verses.tsv")

    def get_trg_draft_file_path(self) -> Path:
        return self._trg_draft_file_path

    @staticmethod
    def _get_trg_draft_file_path_from_confidence_path(confidence_file_path: Path) -> Path:
        return confidence_file_path.with_name(confidence_file_path.name.removesuffix(CONFIDENCE_SUFFIX))

    @abstractmethod
    def generate_confidence_files(
        self,
        translated_draft: TranslatedDraft,
        scripture_refs: Optional[List[ScriptureRef]] = None,
    ) -> None:
        pass

    @abstractmethod
    def _parse_verse_key(self, raw_key: str) -> TVerseKey:
        pass

    def verse_confidence_iterator(self) -> Generator[Tuple[TVerseKey, float], None, None]:
        with open(self.get_verses_path(), "r", encoding="utf-8") as f:
            headers = f.readline().strip().split("\t")
            confidence_index = headers.index("Confidence")
            for line in f:
                cols = line.strip().split("\t")
                vref_or_index = cols[0]
                confidence = float(cols[confidence_index])
                yield (self._parse_verse_key(vref_or_index), confidence)

    def get_verse_confidences(self) -> List[Tuple[TVerseKey, float]]:
        return list(self.verse_confidence_iterator())


class UsfmConfidenceFile(ConfidenceFile[VerseRef]):

    def __init__(self, path: Path):
        super().__init__(path)
        self._book_confidences_cache: Optional[Dict[str, float]] = None

    def get_chapters_path(self) -> Path:
        return self._path.with_suffix(".chapters.tsv")

    def get_books_path(self) -> Path:
        return self._path.parent / "confidences.books.tsv"

    @property
    def _book_confidences(self) -> Dict[str, float]:
        if self._book_confidences_cache is None:
            self._book_confidences_cache = {}
            if self.get_books_path().is_file():
                for book, confidence in self._book_confidence_iterator():
                    self._book_confidences_cache[book] = confidence
        return self._book_confidences_cache

    def _parse_verse_key(self, raw_key: str) -> VerseRef:
        return VerseRef.from_string(raw_key)

    def generate_confidence_files(
        self,
        translated_draft: TranslatedDraft,
        scripture_refs: Optional[List[ScriptureRef]] = None,
    ) -> None:
        if scripture_refs is None:
            raise ValueError("scripture_refs should not be None when generating confidence files for USFM/SFM files.")
        translated_draft.write_confidence_scores_to_file(self._path, scripture_refs)
        translated_draft.write_verse_confidence_scores_to_file(self.get_verses_path(), scripture_refs)
        self.write_chapter_confidence_scores_to_file(translated_draft, scripture_refs)
        self.write_book_confidence_score_to_file(translated_draft, scripture_refs)

    def write_chapter_confidence_scores_to_file(
        self, translated_draft: TranslatedDraft, scripture_refs: List[ScriptureRef]
    ) -> None:
        chapter_confidences: DefaultDict[int, List[float]] = defaultdict(list)
        for vref, confidence in zip(scripture_refs, translated_draft.get_all_sequence_confidence_scores()):
            if not vref.is_verse or confidence is None:
                continue
            chapter_confidences[vref.chapter_num].append(confidence)
        with self.get_chapters_path().open("w", encoding="utf-8", newline="\n") as chapter_confidences_file:
            chapter_confidences_file.write("Chapter\tConfidence\n")
            for chapter, confidences in chapter_confidences.items():
                chapter_confidence = gmean(confidences)
                chapter_confidences_file.write(f"{chapter}\t{chapter_confidence}\n")

    def chapter_confidence_iterator(self) -> Generator[Tuple[int, float], None, None]:
        with open(self.get_chapters_path(), "r", encoding="utf-8") as f:
            headers = f.readline().strip().split("\t")
            confidence_index = headers.index("Confidence")
            for line in f:
                cols = line.strip().split("\t")
                chapter = int(cols[0])
                confidence = float(cols[confidence_index])
                yield (chapter, confidence)

    def get_chapter_confidences(self) -> List[Tuple[int, float]]:
        return list(self.chapter_confidence_iterator())

    def write_book_confidence_score_to_file(
        self, translated_draft: TranslatedDraft, scripture_refs: List[ScriptureRef]
    ) -> None:
        book_confidences: List[float] = []
        for vref, confidence in zip(scripture_refs, translated_draft.get_all_sequence_confidence_scores()):
            if not vref.is_verse or confidence is None:
                continue
            book_confidences.append(confidence)

        current_book = scripture_refs[0].book
        self._book_confidences[current_book] = gmean(book_confidences)
        with self.get_books_path().open("w", encoding="utf-8", newline="\n") as book_confidences_file:
            book_confidences_file.write("Book\tConfidence\n")
            for book, confidence in self._book_confidences.items():
                book_confidences_file.write(f"{book}\t{confidence}\n")

    def _book_confidence_iterator(self) -> Generator[Tuple[str, float], None, None]:
        with open(self.get_books_path(), "r", encoding="utf-8") as f:
            headers = f.readline().strip().split("\t")
            confidence_index = headers.index("Confidence")
            for line in f:
                cols = line.strip().split("\t")
                book = cols[0]
                confidence = float(cols[confidence_index])
                yield (book, confidence)

    def get_book_confidence(self, book: str) -> Optional[float]:
        return self._book_confidences.get(book)


class TxtConfidenceFile(ConfidenceFile[int]):

    def get_files_path(self) -> Path:
        return self._path.parent / "confidences.files.tsv"

    def _parse_verse_key(self, raw_key: str) -> int:
        return int(raw_key)

    def generate_confidence_files(
        self,
        translated_draft: TranslatedDraft,
        scripture_refs: Optional[List[ScriptureRef]] = None,
    ) -> None:
        translated_draft.write_confidence_scores_to_file(self._path)
        translated_draft.write_verse_confidence_scores_to_file(self.get_verses_path())
        self._write_file_confidence_score_to_file(translated_draft)

    def _write_file_confidence_score_to_file(
        self,
        translated_draft: TranslatedDraft,
    ) -> None:
        existing_files: Dict[str, float] = {}
        if self.get_files_path().exists():
            for file_stem, confidence in self.file_confidence_iterator():
                existing_files[file_stem] = confidence

        existing_files[self._trg_draft_file_path.stem] = gmean(
            translated_draft.get_all_sequence_confidence_scores(exclude_none_type=True)
        )
        with self.get_files_path().open("w", encoding="utf-8", newline="\n") as file_confidences_file:
            file_confidences_file.write("File\tConfidence\n")
            for file_stem, confidence in existing_files.items():
                file_confidences_file.write(f"{file_stem}\t{confidence}\n")

    def file_confidence_iterator(self) -> Generator[Tuple[str, float], None, None]:
        with open(self.get_files_path(), "r", encoding="utf-8") as f:
            headers = f.readline().strip().split("\t")
            confidence_index = headers.index("Confidence")
            for line in f:
                cols = line.strip().split("\t")
                file_stem = cols[0]
                confidence = float(cols[confidence_index])
                yield (file_stem, confidence)


class TestConfidenceFile(ConfidenceFile[int]):
    def get_verses_path(self) -> Path:
        # Use the verse-level scores file created by the test script
        return self._path.with_suffix(".scores.tsv")

    def _parse_verse_key(self, raw_key: str) -> int:
        return int(raw_key)

    def generate_confidence_files(
        self,
        translated_draft: TranslatedDraft,
        scripture_refs: Optional[List[ScriptureRef]] = None,
    ) -> None:
        translated_draft.write_confidence_scores_to_file(self._path)


def generate_confidence_files(
    translated_draft: TranslatedDraft,
    trg_draft_file_path: Path,
    scripture_refs: Optional[List[ScriptureRef]] = None,
) -> None:
    if not translated_draft.has_sequence_confidence_scores():
        LOGGER.warning(
            f"{trg_draft_file_path} was not translated with beam search, "
            f"so confidence scores will not be calculated for this file."
        )
        return

    confidence_file = ConfidenceFile.from_draft_file_path(trg_draft_file_path)
    confidence_file.generate_confidence_files(translated_draft, scripture_refs)
