from collections.abc import Set
from typing import List, TypeVar

from machine.corpora import FileParatextProjectSettingsParser, ScriptureRef, UsfmFileText

from silnlp.common.environment import SIL_NLP_ENV

from .passage import Verse, VerseCollector

VerseCollectorType = TypeVar("VerseCollectorType", bound=VerseCollector)


class ParatextProjectReader:
    def __init__(self, project_name: str):
        self._project_name = project_name

    def collect_verses(self, verse_collectors: List[VerseCollectorType]) -> List[VerseCollectorType]:
        settings = FileParatextProjectSettingsParser(SIL_NLP_ENV.pt_projects_dir / self._project_name).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        for book in self._get_all_required_books(verse_collectors):
            usfm_text = UsfmFileText(
                stylesheet,
                encoding,
                book,
                SIL_NLP_ENV.pt_projects_dir / self._project_name / settings.get_book_file_name(book),
            )
            for row in usfm_text:
                for verse_collector in verse_collectors:
                    if isinstance(row.ref, ScriptureRef) and verse_collector.is_ref_in_range(row.ref.verse_ref):
                        verse_collector.add_verse(Verse(row.ref.verse_ref, row.text))
        return verse_collectors

    def _get_all_required_books(self, verse_collectors: List[VerseCollectorType]) -> Set[str]:
        all_books: Set[str] = set()
        for verse_collector in verse_collectors:
            all_books.add(verse_collector.start_ref.book)
            all_books.add(verse_collector.end_ref.book)
        return all_books
