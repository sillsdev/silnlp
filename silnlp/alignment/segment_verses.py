import argparse
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

from machine.corpora import FileParatextProjectSettingsParser, ScriptureRef, UsfmFileText
from machine.scripture import VerseRef
from machine.translation import WordAlignmentMatrix

from silnlp.common.environment import SIL_NLP_ENV

from ..common.corpus import load_corpus, write_corpus
from .eflomal import to_word_alignment_matrix
from .utils import compute_alignment_scores


@dataclass
class Passage:
    start_ref: VerseRef
    end_ref: VerseRef
    text: str


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


class ParallelPassage:
    def __init__(
        self,
        start_ref: VerseRef,
        end_ref: VerseRef,
        source_verses: List[str],
        target_text: str,
        word_alignment_matrix: Optional[WordAlignmentMatrix] = None,
    ):
        self._start_ref = start_ref
        self._end_ref = end_ref
        self._source_verses = source_verses
        self._target_text = target_text
        self._word_alignment_matrix = word_alignment_matrix

    def is_ref_in_range(self, ref: VerseRef) -> bool:
        return ref.compare_to(self._start_ref) >= 0 and ref.compare_to(self._end_ref) <= 0

    def add_source_verse(self, source_verse: str) -> None:
        self._source_verses.append(source_verse)

    def get_source_text(self) -> str:
        return " ".join(self._source_verses)

    def get_target_text(self) -> str:
        return self._target_text

    def load_word_alignment_matrix(self, alignment_line: str) -> None:
        self._word_alignment_matrix = to_word_alignment_matrix(alignment_line)


class ParallelPassageCollection:
    def __init__(self, source_project_name: str, target_passage_file: Path):
        target_passages = PassageReader(target_passage_file).get_passages()
        self._collect_parallel_passages(source_project_name, target_passages)
        self._create_alignments()

    def _get_all_books_in_passages(self, passages: List[Passage]) -> Set[str]:
        all_books: Set[str] = set()
        for passage in passages:
            all_books.add(passage.start_ref.book)
            all_books.add(passage.end_ref.book)
        return all_books

    def _collect_parallel_passages(self, source_project_name: str, target_passages: List[Passage]) -> None:
        self._parallel_passages = [ParallelPassage(t.start_ref, t.end_ref, [], t.text) for t in target_passages]

        settings = FileParatextProjectSettingsParser(SIL_NLP_ENV.pt_projects_dir / source_project_name).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        for book in self._get_all_books_in_passages(target_passages):
            usfm_text = UsfmFileText(
                stylesheet,
                encoding,
                book,
                SIL_NLP_ENV.pt_projects_dir / source_project_name / settings.get_book_file_name(book),
            )
            for row in usfm_text:
                for parallel_passage in self._parallel_passages:
                    if isinstance(row.ref, ScriptureRef) and parallel_passage.is_ref_in_range(row.ref.verse_ref):
                        parallel_passage.add_source_verse(row.text)

    def _create_alignments(self) -> None:
        src_passages = [passage.get_source_text() for passage in self._parallel_passages]
        trg_passages = [passage.get_target_text() for passage in self._parallel_passages]

        with TemporaryDirectory() as td:
            align_path = Path(td, "sym-align.txt")
            write_corpus(Path(td, "src_align.txt"), src_passages)
            write_corpus(Path(td, "trg_align.txt"), trg_passages)
            compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), "fast_align", align_path)

            for index, line in enumerate(load_corpus(align_path)):
                self._parallel_passages[index].load_word_alignment_matrix(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect verse counts and compute alignment scores")
    parser.add_argument("--source-project", help="Name of source Paratext project", required=True, type=str)
    parser.add_argument("--target-passages", help=".tsv file with target passages", required=True, type=str)
    # parser.add_argument(
    #    "--clearml-queue",
    #    default=None,
    #    type=str,
    #    help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
    #    + "it locally and register it with ClearML.",
    # )
    args = parser.parse_args()

    parallel_passages = ParallelPassageCollection(args.source_project, Path(args.target_passages))


if __name__ == "__main__":
    main()
