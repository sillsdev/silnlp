import argparse
from pathlib import Path
from typing import List, Optional

from machine.corpora import FileParatextProjectSettingsParser, ScriptureRef, UsfmFileText
from machine.scripture import VerseRef

from silnlp.common.corpus import load_corpus
from silnlp.common.environment import SIL_NLP_ENV


class Passage:
    start_ref: VerseRef
    end_ref: VerseRef
    verses: Optional[List[str]] = None

    def __init__(self, start_ref: VerseRef, end_ref: VerseRef):
        self.start_ref = start_ref
        self.end_ref = end_ref
        self.verses = []

    def is_ref_in_range(self, ref: VerseRef) -> bool:
        return ref.compare_to(self.start_ref) >= 0 and ref.compare_to(self.end_ref) <= 0

    def join_verses(self) -> str:
        if self.verses is None:
            return ""
        return " ".join([verse for verse in self.verses]).replace("  ", " ")


class PassageReader:
    def __init__(self, passage_file: Path):
        self._passages = self._read_passage_file(passage_file)

    def _read_passage_file(self, passage_file: Path) -> List[Passage]:
        passages = []
        for line in load_corpus(passage_file):
            row = line.split("\t")
            if len(row) < 6:
                continue  # skip malformed lines
            passage = Passage(
                VerseRef(row[0], int(row[1]), int(row[2])),
                VerseRef(row[3], int(row[4]), int(row[5])),
            )
            passages.append(passage)
        return passages

    def get_passages(self) -> List[Passage]:
        return self._passages


def assign_verses_to_passages(project_name: str, passage_file: Path) -> List[Passage]:
    passages = PassageReader(passage_file).get_passages()
    settings = FileParatextProjectSettingsParser(SIL_NLP_ENV.pt_projects_dir / project_name).parse()
    stylesheet = settings.stylesheet
    encoding = settings.encoding
    for passage in passages:
        usfm_text = UsfmFileText(
            stylesheet,
            encoding,
            passage.start_ref.book,
            SIL_NLP_ENV.pt_projects_dir / project_name / settings.get_book_file_name(passage.start_ref.book),
        )
        for row in usfm_text:
            if isinstance(row.ref, ScriptureRef) and passage.is_ref_in_range(row.ref.verse_ref):
                passage.verses.append(row.text)
    return passages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collects text from a Paratext project and groups in into specified passages."
    )
    parser.add_argument("--project", help="Name of source Paratext project", required=True, type=str)
    parser.add_argument(
        "--input-passages", help="Input .tsv file with source passage references", required=True, type=str
    )
    parser.add_argument(
        "--output-passages", help="Output .tsv file to contain target passages", required=True, type=str
    )
    args = parser.parse_args()

    passages = assign_verses_to_passages(args.project, Path(args.input_passages))
    with open(Path(args.output_passages), "w", encoding="utf-8") as out_file:
        for passage in passages:
            out_file.write(
                f"{passage.start_ref.book}\t{passage.start_ref.chapter}\t{passage.start_ref.verse}\t"
                f"{passage.end_ref.book}\t{passage.end_ref.chapter}\t{passage.end_ref.verse}\t"
                f"{passage.join_verses()}\n"
            )


if __name__ == "__main__":
    main()
