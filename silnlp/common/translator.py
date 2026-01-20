import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import AbstractContextManager
from datetime import date
from itertools import groupby
from math import exp
from pathlib import Path
from typing import DefaultDict, Generator, Iterable, List, Optional, Tuple, cast

import docx
import nltk
from iso639 import Lang
from machine.corpora import (
    FileParatextProjectSettingsParser,
    FileParatextProjectTextUpdater,
    ParatextProjectSettings,
    ScriptureRef,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
    UsfmFileText,
    UsfmStylesheet,
    UsfmTextType,
    parse_usfm,
)
from machine.scripture import VerseRef, is_book_id_valid
from scipy.stats import gmean

from silnlp.common.utils import add_tags_to_sentence
from silnlp.nmt.corpora import CorpusPair

from .corpus import load_corpus, write_corpus
from .paratext import get_book_path, get_iso, get_project_dir
from .postprocesser import NoDetectedQuoteConventionException, PostprocessHandler, UnknownQuoteConventionException
from .usfm_utils import PARAGRAPH_TYPE_EMBEDS

LOGGER = logging.getLogger((__package__ or "") + ".translate")
nltk.download("punkt")

CONFIDENCE_SUFFIX = ".confidences.tsv"


# A single translation of a single sentence
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

    def get_translation(self) -> str:
        return self._translation

    def has_sequence_confidence_score(self) -> bool:
        return self._sequence_score is not None

    def get_sequence_confidence_score(self) -> Optional[float]:
        return exp(self._sequence_score) if self._sequence_score is not None else None

    def join_tokens_for_test_file(self) -> str:
        # The first token is skipped because it is always equal to decoder_start_token_id,
        # because of the way Huggingface implements encoder-decoder models
        return " ".join([token for token in self._tokens[1:] if token != "<pad>"])

    def join_tokens_for_confidence_file(self) -> str:
        return "\t".join(self._tokens)

    def join_token_scores_for_confidence_file(self) -> str:
        return "\t".join([str(exp(ts)) for ts in [self._sequence_score] + self._token_scores if ts is not None])


# A group of multiple translations of a single sentence
SentenceTranslationGroup = List[SentenceTranslation]


# A class representing a single draft (one translation of each input sentence)
class TranslatedDraft:
    def __init__(self, sentence_translations: List[SentenceTranslation]):
        self._sentence_translations = sentence_translations

    def has_sequence_confidence_scores(self) -> bool:
        return any([st.has_sequence_confidence_score() for st in self._sentence_translations])

    def write_confidence_scores_to_file(
        self,
        confidences_path: Path,
        row1col1_label: str,
        scripture_refs: Optional[List[ScriptureRef]] = None,
    ) -> None:
        with confidences_path.open("w", encoding="utf-8", newline="\n") as confidences_file:
            confidences_file.write("\t".join([f"{row1col1_label}"] + [f"Token {i}" for i in range(200)]) + "\n")
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
        self, verse_confidences_path: Path, row1col1_label: str, scripture_refs: List[ScriptureRef] = None
    ):
        with verse_confidences_path.open("w", encoding="utf-8", newline="\n") as verse_confidences_file:
            verse_confidences_file.write(f"{row1col1_label}\tConfidence\n")
            for sentence_num, sentence_translation in enumerate(self._sentence_translations):
                if scripture_refs is not None:
                    vref = scripture_refs[sentence_num]
                    if not vref.is_verse:
                        continue
                    label = str(vref)
                else:
                    label = str(sentence_num + 1)
                vref_confidence: Optional[float] = sentence_translation.get_sequence_confidence_score()
                if vref_confidence is not None:
                    verse_confidences_file.write(f"{label}\t{vref_confidence}\n")

    def write_chapter_confidence_scores_to_file(
        self, chapter_confidences_path: Path, scripture_refs: List[ScriptureRef]
    ):
        chapter_confidences: DefaultDict[int, List[float]] = defaultdict(list)
        for sentence_num, vref in enumerate(scripture_refs):
            vref_confidence: Optional[float] = self._sentence_translations[sentence_num].get_sequence_confidence_score()
            if not vref.is_verse or vref_confidence is None:
                continue
            chapter_confidences[vref.chapter_num].append(vref_confidence)

        with chapter_confidences_path.open("w", encoding="utf-8", newline="\n") as chapter_confidences_file:
            chapter_confidences_file.write("Chapter\tConfidence\n")
            for chapter, confidences in chapter_confidences.items():
                chapter_confidence = gmean(confidences)
                chapter_confidences_file.write(f"{chapter}\t{chapter_confidence}\n")

    def get_all_sequence_confidence_scores(self) -> List[float]:
        return [
            scs for scs in [t.get_sequence_confidence_score() for t in self._sentence_translations] if scs is not None
        ]

    def get_all_translations(self) -> List[str]:
        return [st.get_translation() for st in self._sentence_translations]

    def get_all_tokenized_translations(self) -> List[str]:
        return [st.join_tokens_for_test_file() for st in self._sentence_translations]


# A wrapper around List[SentenceTranslationGroup] that allows upstream consumers to view a
# list of translation groups as a collection of discrete drafts
class DraftGroup:
    def __init__(self, translation_groups: List[SentenceTranslationGroup]):
        self.translation_groups = translation_groups
        self.num_drafts: int = len(self.translation_groups[0])

    def get_drafts(self) -> List[TranslatedDraft]:
        translated_draft_sentences: List[List[SentenceTranslation]] = [[] for _ in range(self.num_drafts)]

        for translation_group in self.translation_groups:
            for draft_index in range(self.num_drafts):
                translated_draft_sentences[draft_index].append(translation_group[draft_index])

        return [TranslatedDraft(sentences) for sentences in translated_draft_sentences]


class ConfidenceFile:

    def __init__(self, path: Path, trg_file_path: Optional[Path] = None):
        if trg_file_path is not None:
            self._trg_file_path = trg_file_path
        else:
            self._trg_file_path = path.with_name(path.name.removesuffix(CONFIDENCE_SUFFIX))
        self.path = path

    @classmethod
    def from_trg_path(cls, trg_file_path: Path) -> "ConfidenceFile":
        path = trg_file_path.with_suffix(f"{trg_file_path.suffix}{CONFIDENCE_SUFFIX}")
        return cls(path, trg_file_path)

    def get_path(self) -> Path:
        return self.path

    def get_verses_path(self) -> Path:
        return self.path.with_suffix(".verses.tsv")

    def get_chapters_path(self) -> Path:
        return self.path.with_suffix(".chapters.tsv")

    def get_books_path(self) -> Path:
        return self.path.parent / "confidences.books.tsv"

    def get_files_path(self, trg_prefix: str = "") -> Path:
        return self.path.parent / f"{trg_prefix}confidences.files.tsv"

    def generate_usfm_confidence_files(
        self,
        translated_draft: TranslatedDraft,
        scripture_refs: List[ScriptureRef],
    ) -> None:
        translated_draft.write_confidence_scores_to_file(self.path, "VRef", scripture_refs)
        translated_draft.write_verse_confidence_scores_to_file(self.get_verses_path(), "VRef", scripture_refs)
        translated_draft.write_chapter_confidence_scores_to_file(self.get_chapters_path(), scripture_refs)
        self._append_book_confidence_score(translated_draft, scripture_refs)

    def _append_book_confidence_score(
        self,
        translated_draft: TranslatedDraft,
        scripture_refs: List[ScriptureRef],
    ) -> None:
        book_confidences_path = self.get_books_path()
        row1_col1_header = "Book"
        if scripture_refs:
            col1_entry = scripture_refs[0].book
        else:
            col1_entry = self._trg_file_path.stem

        with book_confidences_path.open("a", encoding="utf-8", newline="\n") as book_confidences_file:
            if book_confidences_file.tell() == 0:
                book_confidences_file.write(f"{row1_col1_header}\tConfidence\n")
            book_confidences_file.write(
                f"{col1_entry}\t{gmean(translated_draft.get_all_sequence_confidence_scores())}\n"
            )

    def generate_txt_confidence_files(
        self,
        translated_draft: TranslatedDraft,
        trg_prefix: str = "",
    ) -> None:
        translated_draft.write_confidence_scores_to_file(self.path, "Sequence Number")
        translated_draft.write_verse_confidence_scores_to_file(self.path.with_suffix(".verses.tsv"), "Sequence Number")
        self._append_file_confidence_score(translated_draft, trg_prefix)

    def _append_file_confidence_score(
        self,
        translated_draft: TranslatedDraft,
        trg_prefix: str = "",
    ) -> None:
        file_confidences_path = self.get_files_path(trg_prefix)

        with file_confidences_path.open("a", encoding="utf-8", newline="\n") as file_confidences_file:
            if file_confidences_file.tell() == 0:
                file_confidences_file.write("File\tConfidence\n")
            file_confidences_file.write(
                f"{self._trg_file_path.stem}\t{gmean(translated_draft.get_all_sequence_confidence_scores())}\n"
            )


def generate_confidence_files(
    translated_draft: TranslatedDraft,
    trg_file_path: Path,
    scripture_refs: Optional[List[ScriptureRef]] = None,
) -> None:
    if not translated_draft.has_sequence_confidence_scores():
        LOGGER.warning(
            f"{trg_file_path} was not translated with beam search, so confidence scores will not be calculated for this file."
        )
        return

    confidence_file = ConfidenceFile.from_trg_path(trg_file_path)

    ext = trg_file_path.suffix.lower()
    if ext in {".usfm", ".sfm"}:
        assert scripture_refs is not None
        confidence_file.generate_usfm_confidence_files(translated_draft, scripture_refs)
    elif ext == ".txt":
        confidence_file.generate_txt_confidence_files(translated_draft)
    else:
        raise ValueError(
            f"Invalid trg file extension {ext} when using --save-confidences in the translate step."
            f"Valid file extensions for --save-confidences are .usfm, .sfm, and .txt."
        )


def generate_test_confidence_files(
    translated_draft: TranslatedDraft,
    trg_file_path: Path,
) -> None:
    confidence_file = ConfidenceFile.from_trg_path(trg_file_path)
    translated_draft.write_confidence_scores_to_file(confidence_file.get_path(), "Sequence Number")


class Translator(AbstractContextManager["Translator"], ABC):
    @abstractmethod
    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        vrefs: Optional[Iterable[VerseRef]] = None,
    ) -> Generator[SentenceTranslationGroup, None, None]:
        pass

    def translate_text(
        self,
        src_file_path: Path,
        trg_file_path: Path,
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        trg_prefix: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:

        sentences = [add_tags_to_sentence(tags, sentence) for sentence in load_corpus(src_file_path)]
        sentence_translation_groups: List[SentenceTranslationGroup] = list(
            self.translate(sentences, src_iso, trg_iso, produce_multiple_translations)
        )
        draft_set = DraftGroup(sentence_translation_groups)
        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            if produce_multiple_translations:
                trg_draft_file_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
            else:
                trg_draft_file_path = trg_file_path
            write_corpus(trg_draft_file_path, translated_draft.get_all_translations())

            if save_confidences:
                generate_confidence_files(
                    translated_draft,
                    trg_draft_file_path,
                )

    def translate_book(
        self,
        src_project: str,
        book: str,
        output_path: Path,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        chapters: List[int] = [],
        trg_project: Optional[str] = None,
        postprocess_handler: PostprocessHandler = PostprocessHandler(),
        experiment_ckpt_str: str = "",
        training_corpus_pairs: List[CorpusPair] = [],
        tags: Optional[List[str]] = None,
    ) -> None:
        book_path = get_book_path(src_project, book)
        if not book_path.is_file():
            raise RuntimeError(f"Can't find file {book_path} for book {book}")
        else:
            LOGGER.info(f"Found the file {book_path} for book {book}")

        self.translate_usfm(
            book_path,
            output_path,
            get_iso(get_project_dir(src_project)),
            trg_iso,
            produce_multiple_translations,
            save_confidences,
            chapters,
            trg_project,
            postprocess_handler,
            experiment_ckpt_str,
            training_corpus_pairs,
            tags,
        )

    def translate_usfm(
        self,
        src_file_path: Path,
        trg_file_path: Path,
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        save_confidences: bool = False,
        chapters: List[int] = [],
        trg_project: Optional[str] = None,
        postprocess_handler: PostprocessHandler = PostprocessHandler(),
        experiment_ckpt_str: str = "",
        training_corpus_pairs: List[CorpusPair] = [],
        tags: Optional[List[str]] = None,
    ) -> None:
        # Create UsfmFileText object for source
        src_from_project = False
        src_settings: Optional[ParatextProjectSettings] = None
        stylesheet = UsfmStylesheet("usfm.sty")
        if str(src_file_path).startswith(str(get_project_dir(""))):
            src_from_project = True
            src_settings = FileParatextProjectSettingsParser(src_file_path.parent).parse()
            stylesheet = src_settings.stylesheet
            book_id = src_settings.get_book_id(src_file_path.name)
            assert book_id is not None

            src_file_text = UsfmFileText(
                src_settings.stylesheet,
                src_settings.encoding,
                book_id,
                src_file_path,
                src_settings.versification,
                include_all_text=True,
                project=src_settings.name,
            )
        else:
            # Guess book ID
            with src_file_path.open(encoding="utf-8-sig") as f:
                book_id = f.read().split()[1].upper()
            if not is_book_id_valid(book_id):
                raise ValueError(f"Book ID not detected: {book_id}")

            src_file_text = UsfmFileText(stylesheet, "utf-8-sig", book_id, src_file_path, include_all_text=True)

        sentences = [re.sub(" +", " ", add_tags_to_sentence(tags, s.text.strip())) for s in src_file_text]
        scripture_refs: List[ScriptureRef] = [s.ref for s in src_file_text]
        vrefs: List[VerseRef] = [sr.verse_ref for sr in scripture_refs]
        LOGGER.info(f"File {src_file_path} parsed correctly.")

        # Filter sentences
        for i in reversed(range(len(sentences))):
            marker = scripture_refs[i].path[-1].name if len(scripture_refs[i].path) > 0 else ""
            if (
                (len(chapters) > 0 and scripture_refs[i].chapter_num not in chapters)
                or marker in PARAGRAPH_TYPE_EMBEDS
                or stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT
            ):
                sentences.pop(i)
                scripture_refs.pop(i)
        empty_sents: List[Tuple[int, ScriptureRef]] = []
        for i in reversed(range(len(sentences))):
            if len(sentences[i].strip()) == 0:
                sentences.pop(i)
                empty_sents.append((i, scripture_refs.pop(i)))

        sentence_translation_groups: List[SentenceTranslationGroup] = list(
            self.translate(sentences, src_iso, trg_iso, produce_multiple_translations, vrefs)
        )
        num_drafts = len(sentence_translation_groups[0])

        # Add empty sentences back in
        # Prevents pre-existing text from showing up in the sections of translated text
        for idx, vref in reversed(empty_sents):
            sentences.insert(idx, "")
            scripture_refs.insert(idx, vref)
            sentence_translation_groups.insert(idx, [SentenceTranslation("", [], [], None)] * num_drafts)

        text_behavior = (
            UpdateUsfmTextBehavior.PREFER_NEW if trg_project is not None else UpdateUsfmTextBehavior.STRIP_EXISTING
        )

        draft_set: DraftGroup = DraftGroup(sentence_translation_groups)
        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            postprocess_handler.construct_rows(scripture_refs, sentences, translated_draft.get_all_translations())

            for config in postprocess_handler.configs:

                # Compile draft remarks
                draft_src_str = f"project {src_file_text.project}" if src_from_project else f"file {src_file_path.name}"
                draft_remark = f"This draft of {scripture_refs[0].book} was machine translated on {date.today()} from {draft_src_str} using model {experiment_ckpt_str}. It should be reviewed and edited carefully."
                postprocess_remark = config.get_postprocess_remark()
                remarks = [draft_remark] + ([postprocess_remark] if postprocess_remark else [])

                # Insert translation into the USFM structure of an existing project
                # If the target project is not the same as the translated file's original project,
                # no verses outside of the ones translated will be overwritten
                if trg_project is not None or src_from_project:
                    project_dir = get_project_dir(trg_project if trg_project is not None else src_file_path.parent.name)
                    dest_updater = FileParatextProjectTextUpdater(project_dir)
                    usfm_out = dest_updater.update_usfm(
                        book_id=src_file_text.id,
                        rows=config.rows,
                        text_behavior=text_behavior,
                        paragraph_behavior=config.get_paragraph_behavior(),
                        embed_behavior=config.get_embed_behavior(),
                        style_behavior=config.get_style_behavior(),
                        update_block_handlers=(
                            config.create_place_markers_postprocessor().get_update_block_handlers()
                            if config.is_marker_placement_required()
                            else None
                        ),
                        remarks=remarks,
                        compare_segments=True,
                    )

                    if usfm_out is None:
                        raise FileNotFoundError(
                            f"Book {src_file_text.id} does not exist in target project {trg_project}"
                        )
                else:  # Slightly more manual version for updating an individual file
                    with open(src_file_path, encoding="utf-8-sig") as f:
                        usfm = f.read()
                    handler = UpdateUsfmParserHandler(
                        rows=config.rows,
                        id_text=scripture_refs[0].book,
                        text_behavior=text_behavior,
                        paragraph_behavior=config.get_paragraph_behavior(),
                        embed_behavior=config.get_embed_behavior(),
                        style_behavior=config.get_style_behavior(),
                        update_block_handlers=(
                            config.create_place_markers_postprocessor().get_update_block_handlers()
                            if config.is_marker_placement_required()
                            else None
                        ),
                        remarks=remarks,
                    )
                    parse_usfm(usfm, handler)
                    usfm_out = handler.get_usfm()

                if config.is_quotation_mark_denormalization_required():
                    try:
                        quotation_denormalization_postprocessor = (
                            config.create_denormalize_quotation_marks_postprocessor(training_corpus_pairs)
                        )
                        usfm_out = quotation_denormalization_postprocessor.postprocess_usfm(usfm_out)
                    except (UnknownQuoteConventionException, NoDetectedQuoteConventionException) as e:
                        LOGGER.warning(str(e) + " Skipping quotation mark denormalization.")
                        continue

                # Construct output file name write to file
                trg_draft_file_path = trg_file_path.with_stem(trg_file_path.stem + config.get_postprocess_suffix())
                if produce_multiple_translations:
                    trg_draft_file_path = trg_draft_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")

                with trg_draft_file_path.open(
                    "w",
                    encoding=(
                        "utf-8"
                        if src_settings is None
                        or src_settings.encoding == "utf-8-sig"
                        or src_settings.encoding == "utf_8_sig"
                        else src_settings.encoding
                    ),
                ) as f:
                    f.write(usfm_out)

            if save_confidences:
                generate_confidence_files(
                    translated_draft,
                    trg_file_path,
                    produce_multiple_translations=produce_multiple_translations,
                    scripture_refs=scripture_refs,
                    draft_index=draft_index,
                )

    def translate_docx(
        self,
        src_file_path: Path,
        trg_file_path: Path,
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        tags: Optional[List[str]] = None,
    ) -> None:
        tokenizer: nltk.tokenize.PunktSentenceTokenizer
        try:
            src_lang = Lang(src_iso)
            tokenizer = nltk.data.load(f"tokenizers/punkt/{src_lang.name.lower()}.pickle")
        except Exception:
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

        with src_file_path.open("rb") as file:
            doc = docx.Document(file)

        sentences: List[str] = []
        paras: List[int] = []

        for i, paragraph in enumerate(doc.paragraphs):
            for sentence in tokenizer.tokenize(paragraph.text):
                sentences.append(add_tags_to_sentence(tags, sentence))
                paras.append(i)

        draft_set: DraftGroup = DraftGroup(
            list(self.translate(sentences, src_iso, trg_iso, produce_multiple_translations))
        )

        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            for para, group in groupby(zip(translated_draft.get_all_translations(), paras), key=lambda t: t[1]):
                text = " ".join(s[0] for s in group)
                doc.paragraphs[para].text = text

            if produce_multiple_translations:
                trg_draft_file_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
            else:
                trg_draft_file_path = trg_file_path

            with trg_draft_file_path.open("wb") as file:
                doc.save(file)

    def __enter__(self) -> "Translator":
        return self

    def __exit__(
        self, exc_type, exc_val, exc_tb  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    ) -> None:
        pass
