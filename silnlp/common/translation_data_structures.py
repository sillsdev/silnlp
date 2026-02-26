from math import exp
from pathlib import Path
from statistics import mean
from typing import List, Optional

from machine.corpora import ScriptureRef


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
        self.num_drafts: int = self.translation_groups[0].num_drafts

    def get_drafts(self) -> List[TranslatedDraft]:
        translated_draft_sentences: List[List[SentenceTranslation]] = [[] for _ in range(self.num_drafts)]

        for translation_group in self.translation_groups:
            for draft_index, sentence_translation in enumerate(translation_group):
                translated_draft_sentences[draft_index].append(sentence_translation)

        return [TranslatedDraft(sentences) for sentences in translated_draft_sentences]
