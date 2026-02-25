import re
from pathlib import Path
from typing import List, Optional, Tuple

from attr import dataclass
from machine.corpora import (
    FileParatextProjectSettingsParser,
    ScriptureRef,
    TextRow,
    UsfmFileText,
    UsfmStylesheet,
    UsfmTextType,
    UsfmTokenizer,
    UsfmTokenType,
)
from machine.scripture import VerseRef

from silnlp.common.postprocesser import PostprocessHandler
from silnlp.common.translation_data_structures import (
    DraftGroup,
    SentenceTranslation,
    SentenceTranslationGroup,
    TranslatedDraft,
)
from silnlp.common.utils import add_tags_to_sentence

# Marker "type" is as defined by the UsfmTokenType given to tokens by the UsfmTokenizer,
# which mostly aligns with a marker's StyleType in the USFM stylesheet
CHARACTER_TYPE_EMBEDS = ["fig", "fm", "jmp", "rq", "va", "vp", "xt", "xtSee", "xtSeeAlso"]
PARAGRAPH_TYPE_EMBEDS = ["lit", "r", "rem"]
NON_NOTE_TYPE_EMBEDS = CHARACTER_TYPE_EMBEDS + PARAGRAPH_TYPE_EMBEDS


class UsfmTextRowCollection:
    def __init__(
        self,
        file_text: UsfmFileText,
        stylesheet: UsfmStylesheet,
        selected_chapters: Optional[List[int]] = None,
        tags: Optional[List[str]] = None,
    ):
        self._text_rows = [s for s in file_text]
        self._stylesheet = stylesheet
        self._selected_chapters = selected_chapters
        self._tags = tags

    def get_sentences_and_vrefs_for_translation(self) -> Tuple[List[str], List[VerseRef]]:
        non_empty_rows = self._filter_rows()
        if self._tags is None:
            return ([s.text.strip() for s in non_empty_rows], [s.ref for s in non_empty_rows])
        else:
            return ([self._clean_and_tag_sentence(s.text) for s in non_empty_rows], [s.ref for s in non_empty_rows])

    def _filter_rows(self) -> List[TextRow]:
        return [s for s in self._text_rows if not self._should_exclude_row(s)]

    def _should_exclude_row(self, row: TextRow) -> bool:
        marker = row.ref.path[-1].name if len(row.ref.path) > 0 else ""
        if (
            (self._selected_chapters is not None and row.ref.chapter_num not in self._selected_chapters)
            or marker in PARAGRAPH_TYPE_EMBEDS
            or self._stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT
        ):
            return True
        return False

    def _clean_and_tag_sentence(self, sentence: str) -> str:
        if self._tags is None:
            return sentence.strip()
        else:
            return re.sub(" +", " ", add_tags_to_sentence(self._tags, sentence.strip()))

    def get_book(self) -> str:
        return self._text_rows[0].ref.book

    def to_translated_text_row_collection(
        self, translated_sentences: List[SentenceTranslationGroup]
    ) -> "TranslatedTextRowCollection":
        translated_text_rows = []
        num_drafts = len(translated_sentences[0]) if len(translated_sentences) > 0 else 1
        for text_row in self._text_rows:
            if self._should_exclude_row(text_row):
                translated_sentence = [SentenceTranslation("", [], [], None)] * num_drafts
            else:
                translated_sentence = translated_sentences.pop(0)
            translated_text_rows.append(
                TranslatedTextRow(ref=text_row.ref, text=text_row.text, translated_sentence=translated_sentence)
            )

        if len(translated_text_rows) != len(self._text_rows):
            raise ValueError(
                f"The number of translated sentences ({len(translated_text_rows)}) does not match number of sentences in the document ({len(self._text_rows)})."
            )

        return TranslatedTextRowCollection(translated_text_rows)


@dataclass
class TranslatedTextRow:
    ref: ScriptureRef
    text: str
    translated_sentence: SentenceTranslationGroup


class TranslatedTextRowCollection:
    def __init__(self, rows: List[TranslatedTextRow]):
        self._rows = rows
        self._draft_group = self._create_draft_group()

    def _create_draft_group(self) -> DraftGroup:
        return DraftGroup([r.translated_sentence for r in self._rows])

    def get_translated_drafts(self) -> List[TranslatedDraft]:
        return self._draft_group.get_drafts()

    def construct_postprocessing_rows_for_draft_index(
        self, postprocess_handler: PostprocessHandler, draft_index: int
    ) -> None:
        postprocess_handler.construct_rows(
            [r.ref for r in self._rows],
            [r.text for r in self._rows],
            self._draft_group.get_drafts()[draft_index - 1].get_all_translations(),
        )

    def get_scripture_refs(self) -> List[ScriptureRef]:
        return [r.ref for r in self._rows]


def main() -> None:
    """
    Print out all paragraph and character markers for a book
    To use set book, fpath, and out_path. fpath should be a path to a book in a Paratext project
    """

    book = "MAT"
    fpath = Path("")
    out_path = Path("")
    sentences_file = Path("")

    settings = FileParatextProjectSettingsParser(fpath.parent).parse()
    file_text = UsfmFileText(
        settings.stylesheet,
        settings.encoding,
        book,
        fpath,
        settings.versification,
        include_markers=True,
        include_all_text=True,
        project=settings.name,
    )

    vrefs = []
    usfm_markers = []
    usfm_tokenizer = UsfmTokenizer(settings.stylesheet)
    with sentences_file.open("w", encoding=settings.encoding) as f:
        for sent in file_text:
            f.write(f"{sent}\n")
            if len(sent.ref.path) > 0 and sent.ref.path[-1].name in PARAGRAPH_TYPE_EMBEDS:
                continue

            vrefs.append(sent.ref)
            usfm_markers.append([])
            usfm_toks = usfm_tokenizer.tokenize(sent.text.strip())

            ignore_scope = None
            for tok in usfm_toks:
                if ignore_scope is not None:
                    if tok.type == UsfmTokenType.END and tok.marker[:-1] == ignore_scope.marker:
                        ignore_scope = None
                elif tok.type == UsfmTokenType.NOTE or (
                    tok.type == UsfmTokenType.CHARACTER and tok.marker in CHARACTER_TYPE_EMBEDS
                ):
                    ignore_scope = tok
                elif tok.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHARACTER, UsfmTokenType.END]:
                    usfm_markers[-1].append(tok.marker)

    with out_path.open("w", encoding=settings.encoding) as f:
        for ref, markers in zip(vrefs, usfm_markers):
            f.write(f"{ref} {markers}\n")


if __name__ == "__main__":
    main()
