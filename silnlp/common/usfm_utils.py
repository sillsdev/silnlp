import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from attr import dataclass
from iso639 import Lang
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

from .postprocesser import PostprocessHandler
from .translation_data_structures import DraftGroup, SentenceTranslation, SentenceTranslationGroup, TranslatedDraft
from .utils import NLTKSentenceTokenizer, add_tags_to_sentence

# Marker "type" is as defined by the UsfmTokenType given to tokens by the UsfmTokenizer,
# which mostly aligns with a marker's StyleType in the USFM stylesheet
CHARACTER_TYPE_EMBEDS = ["fig", "fm", "jmp", "rq", "va", "vp", "xt", "xtSee", "xtSeeAlso"]
PARAGRAPH_TYPE_EMBEDS = ["lit", "r", "rem"]
NON_NOTE_TYPE_EMBEDS = CHARACTER_TYPE_EMBEDS + PARAGRAPH_TYPE_EMBEDS


class UsfmTextRowCollection:
    def __init__(
        self,
        file_text: UsfmFileText,
        src_iso: str,
        stylesheet: UsfmStylesheet,
        selected_chapters: Optional[List[int]] = None,
        tags: Optional[List[str]] = None,
    ):
        self._text_rows = [s for s in file_text]
        self._src_iso = src_iso
        self._stylesheet = stylesheet
        self._selected_chapters = selected_chapters
        self._tags = tags

        self._empty_row_indices: Set[int] = self._find_indices_of_empty_rows()
        self._subdivided_row_texts: Dict[int, List[str]] = self._split_non_verse_rows_if_necessary()

    def _find_indices_of_empty_rows(self) -> Set[int]:
        return set([i for i, row in enumerate(self._text_rows) if self._is_row_empty(row)])

    def _is_row_empty(self, row: TextRow) -> bool:
        marker = row.ref.path[-1].name if len(row.ref.path) > 0 else ""
        if (
            (
                self._selected_chapters is not None
                and len(self._selected_chapters) > 0
                and row.ref.chapter_num not in self._selected_chapters
            )
            or marker in PARAGRAPH_TYPE_EMBEDS
            or self._stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT
        ):
            return True
        return False

    def _split_non_verse_rows_if_necessary(self) -> Dict[int, List[str]]:
        subdivided_row_texts: Dict[int, List[str]] = {}
        for i, row in enumerate(self._text_rows):
            if not row.ref.is_verse() and len(row.text) > 200:
                split_sentences = self._split_sentences(row.text)
                if len(split_sentences) > 0:
                    subdivided_row_texts[i] = split_sentences
        return subdivided_row_texts

    def _split_sentences(self, text) -> List[str]:
        return NLTKSentenceTokenizer.for_iso(self._src_iso).tokenize(text)

    def get_sentences_and_vrefs_for_translation(self) -> Tuple[List[str], List[ScriptureRef]]:
        sentences, scripture_refs = self._match_all_sentences_with_scripture_refs()
        return ([self._clean_and_tag_sentence(s) for s in sentences], scripture_refs)

    def _filter_out_empty_rows(self) -> List[TextRow]:
        return [s for i, s in enumerate(self._text_rows) if i not in self._empty_row_indices]

    def _match_all_sentences_with_scripture_refs(self) -> Tuple[List[str], List[ScriptureRef]]:
        sentences: List[str] = []
        scripture_refs: List[ScriptureRef] = []
        for i, row in enumerate(self._text_rows):
            if i in self._empty_row_indices:
                continue
            elif i in self._subdivided_row_texts:
                sentences.extend(self._subdivided_row_texts[i])
                scripture_refs.extend([row.ref] * len(self._subdivided_row_texts[i]))
            else:
                sentences.append(row.text)
                scripture_refs.append(row.ref)
        return sentences, scripture_refs

    def _clean_and_tag_sentence(self, sentence: str) -> str:
        if self._tags is None:
            return sentence.strip()
        else:
            return re.sub(" +", " ", add_tags_to_sentence(self._tags, sentence.strip()))

    def get_book(self) -> str:
        return self._text_rows[0].ref.book

    def to_translated_text_row_collection(
        self, sentence_translation_groups: List[SentenceTranslationGroup]
    ) -> "TranslatedTextRowCollection":
        translated_text_rows = []
        num_drafts = sentence_translation_groups[0].num_drafts if len(sentence_translation_groups) > 0 else 0
        for i, text_row in enumerate(self._text_rows):
            if i in self._empty_row_indices:
                translated_sentence = SentenceTranslationGroup([SentenceTranslation("", [], [], None)] * num_drafts)
            elif i in self._subdivided_row_texts:
                split_translations: List[SentenceTranslationGroup] = []
                for _ in range(len(self._subdivided_row_texts[i])):
                    split_translations.append(sentence_translation_groups.pop(0))
                translated_sentence = SentenceTranslationGroup.combine(split_translations)
            else:
                translated_sentence = sentence_translation_groups.pop(0)
            translated_text_rows.append(TranslatedTextRow(text_row.ref, text_row.text, translated_sentence))

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
