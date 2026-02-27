import re
from math import exp
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Set, Tuple

from attr import dataclass
from machine.corpora import ScriptureRef, TextRow, UsfmFileText, UsfmStylesheet, UsfmTextType

from .postprocesser import PostprocessHandler
from .utils import NLTKSentenceTokenizer, add_tags_to_sentence


class SentenceTranslation:
    def __init__(
        self,
        translation: str,
        tokens: List[str],
        token_scores: List[float],
        sequence_score: Optional[float],
    ):
        self._translation = translation
        self._tokens = tokens
        self._token_scores = token_scores
        self._sequence_score = sequence_score

    @classmethod
    def combine(cls, translations: List["SentenceTranslation"]) -> "SentenceTranslation":
        if len(translations) == 0:
            raise ValueError("Unable to combine an empty list of SentenceTranslations.")

        combined_translation: str = " ".join([t.get_translation() for t in translations])
        combined_tokens: List[str] = [token for t in translations for token in t._tokens]
        combined_token_scores: List[float] = [ts for t in translations for ts in t._token_scores if ts is not None]
        combined_sequence_score: Optional[float] = (
            mean([t._sequence_score for t in translations if t.has_sequence_confidence_score()])
            if all(t.has_sequence_confidence_score() for t in translations)
            else None
        )
        return cls(combined_translation, combined_tokens, combined_token_scores, combined_sequence_score)

    def get_translation(self) -> str:
        return self._translation

    def has_sequence_confidence_score(self) -> bool:
        return self._sequence_score is not None

    def get_sequence_confidence_score(self) -> Optional[float]:
        return exp(self._sequence_score) if self._sequence_score is not None else None

    def join_tokens_for_test_file(self) -> str:
        return " ".join([token for token in self._tokens[1:] if token != "<pad>"])

    def join_tokens_for_confidence_file(self) -> str:
        return "\t".join(self._tokens)

    def join_token_scores_for_confidence_file(self) -> str:
        return "\t".join([str(exp(ts)) for ts in [self._sequence_score] + self._token_scores if ts is not None])


class SentenceTranslationGroup:
    def __init__(self, translations: List[SentenceTranslation]):
        self._translations = translations

    @property
    def num_drafts(self) -> int:
        return len(self._translations)

    @classmethod
    def combine(cls, groups: List["SentenceTranslationGroup"]) -> "SentenceTranslationGroup":
        if len(groups) == 0:
            raise ValueError("Unable to combine an empty list of SentenceTranslationGroups.")

        combined_translations: List[SentenceTranslation] = []
        num_drafts = groups[0].num_drafts
        for n in range(num_drafts):
            combined_translations.append(SentenceTranslation.combine([g._translations[n] for g in groups]))

        return cls(combined_translations)

    def __iter__(self):
        return iter(self._translations)


class TranslatedDraft:
    def __init__(self, sentence_translations: List[SentenceTranslation]):
        self._sentence_translations = sentence_translations

    def has_sequence_confidence_scores(self) -> bool:
        return any([st.has_sequence_confidence_score() for st in self._sentence_translations])

    def write_confidence_scores_to_file(
        self,
        confidences_path: Path,
        scripture_refs: Optional[List[ScriptureRef]] = None,
    ) -> None:
        sequence_id_header = self._get_sequence_id_header_for_confidence_file(scripture_refs)
        with confidences_path.open("w", encoding="utf-8", newline="\n") as confidences_file:
            confidences_file.write("\t".join([f"{sequence_id_header}"] + [f"Token {i}" for i in range(200)]) + "\n")
            confidences_file.write("\t".join(["Sequence Score"] + [f"Token Score {i}" for i in range(200)]) + "\n")
            for sentence_num, sentence_translation in enumerate(self._sentence_translations):
                if not sentence_translation.has_sequence_confidence_score():
                    continue

                if scripture_refs is not None:
                    sequence_label = str(scripture_refs[sentence_num])
                else:
                    sequence_label = str(sentence_num + 1)
                confidences_file.write(
                    sequence_label + "\t" + sentence_translation.join_tokens_for_confidence_file() + "\n"
                )
                confidences_file.write(sentence_translation.join_token_scores_for_confidence_file() + "\n")

    def write_verse_confidence_scores_to_file(
        self, verse_confidences_path: Path, scripture_refs: Optional[List[ScriptureRef]] = None
    ) -> None:
        sequence_id_header = self._get_sequence_id_header_for_confidence_file(scripture_refs)
        with verse_confidences_path.open("w", encoding="utf-8", newline="\n") as verse_confidences_file:
            verse_confidences_file.write(f"{sequence_id_header}\tConfidence\n")
            for sentence_num, confidence in enumerate(self.get_all_sequence_confidence_scores()):
                if scripture_refs is not None:
                    vref = scripture_refs[sentence_num]
                    if not vref.is_verse:
                        continue
                    label = str(vref)
                else:
                    label = str(sentence_num + 1)
                if confidence is not None:
                    verse_confidences_file.write(f"{label}\t{confidence}\n")

    @staticmethod
    def _get_sequence_id_header_for_confidence_file(scripture_refs: Optional[List[ScriptureRef]] = None) -> str:
        if scripture_refs is not None:
            return "VRef"
        return "Sequence Number"

    def get_all_sequence_confidence_scores(self, exclude_none_type: bool = False) -> List[Optional[float]]:
        if exclude_none_type:
            return [
                scs
                for scs in [t.get_sequence_confidence_score() for t in self._sentence_translations]
                if scs is not None
            ]
        return [st.get_sequence_confidence_score() for st in self._sentence_translations]

    def get_all_translations(self) -> List[str]:
        return [st.get_translation() for st in self._sentence_translations]

    def get_all_tokenized_translations(self) -> List[str]:
        return [st.join_tokens_for_test_file() for st in self._sentence_translations]


class DraftGroup:
    def __init__(self, translation_groups: List[SentenceTranslationGroup]):
        self.translation_groups = translation_groups
        self.num_drafts: int = self.translation_groups[0].num_drafts if len(self.translation_groups) > 0 else 0

    def get_drafts(self) -> List[TranslatedDraft]:
        translated_draft_sentences: List[List[SentenceTranslation]] = [[] for _ in range(self.num_drafts)]

        for translation_group in self.translation_groups:
            for draft_index, sentence_translation in enumerate(translation_group):
                translated_draft_sentences[draft_index].append(sentence_translation)

        return [TranslatedDraft(sentences) for sentences in translated_draft_sentences]


class UsfmTextRowCollection:
    _SENTENCE_SPLIT_LENGTH_THRESHOLD = 200

    def __init__(
        self,
        file_text: UsfmFileText,
        src_iso: str,
        stylesheet: UsfmStylesheet,
        selected_chapters: Optional[List[int]] = None,
        tags: Optional[List[str]] = None,
    ):
        self._src_iso = src_iso
        self._stylesheet = stylesheet
        self._selected_chapters = selected_chapters
        self._tags = tags
        self._text_rows = self._skip_unneeded_rows([s for s in file_text])

        self._empty_row_indices: Set[int] = self._find_indices_of_empty_rows()
        self._subdivided_row_texts: Dict[int, List[str]] = self._split_non_verse_rows_if_necessary()

    def _skip_unneeded_rows(self, rows: List[TextRow]) -> List[TextRow]:
        return [row for row in rows if not self._is_row_unneeded(row)]

    def _is_row_unneeded(self, row: TextRow) -> bool:
        marker = row.ref.path[-1].name if len(row.ref.path) > 0 else ""
        if (
            (
                self._selected_chapters is not None
                and len(self._selected_chapters) > 0
                and row.ref.chapter_num not in self._selected_chapters
            )
            or marker in ["lit", "r", "rem"]
            or self._stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT
        ):
            return True
        return False

    def _find_indices_of_empty_rows(self) -> Set[int]:
        return set([i for i, row in enumerate(self._text_rows) if self._is_row_empty(row)])

    def _is_row_empty(self, row: TextRow) -> bool:
        return len(row.text.strip()) == 0

    def _split_non_verse_rows_if_necessary(self) -> Dict[int, List[str]]:
        subdivided_row_texts: Dict[int, List[str]] = {}
        for i, row in enumerate(self._text_rows):
            if not row.ref.is_verse and len(row.text) > self._SENTENCE_SPLIT_LENGTH_THRESHOLD:
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
            return re.sub(" +", " ", sentence.strip())
        else:
            return re.sub(" +", " ", add_tags_to_sentence(self._tags, sentence.strip()))

    def get_book(self) -> str:
        if len(self._text_rows) == 0:
            raise ValueError("Unable to determine book name with no USFM text rows.")
        return self._text_rows[0].ref.book

    def to_translated_text_row_collection(
        self, sentence_translation_groups: List[SentenceTranslationGroup]
    ) -> "TranslatedTextRowCollection":
        translated_text_rows = []
        num_drafts = sentence_translation_groups[0].num_drafts if len(sentence_translation_groups) > 0 else 0
        current_translation_group_index = 0
        for i, text_row in enumerate(self._text_rows):
            if i in self._empty_row_indices:
                translated_sentence = SentenceTranslationGroup([SentenceTranslation("", [], [], None)] * num_drafts)
            elif i in self._subdivided_row_texts:
                split_translations: List[SentenceTranslationGroup] = []
                for _ in range(len(self._subdivided_row_texts[i])):
                    split_translations.append(sentence_translation_groups[current_translation_group_index])
                    current_translation_group_index += 1
                translated_sentence = SentenceTranslationGroup.combine(split_translations)
            else:
                translated_sentence = sentence_translation_groups[current_translation_group_index]
                current_translation_group_index += 1
            translated_text_rows.append(TranslatedTextRow(text_row.ref, text_row.text, translated_sentence))

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
