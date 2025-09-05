import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import date
from itertools import groupby
from math import exp
from pathlib import Path
from typing import DefaultDict, Iterable, List, Optional

import docx
import nltk
from iso639 import Lang
from machine.corpora import (
    FileParatextProjectSettingsParser,
    FileParatextProjectTextUpdater,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
    UsfmFileText,
    UsfmStylesheet,
    UsfmTextType,
    parse_usfm,
)
from machine.scripture import VerseRef, is_book_id_valid
from scipy.stats import gmean

from silnlp.nmt.corpora import CorpusPair

from .corpus import load_corpus, write_corpus
from .paratext import get_book_path, get_iso, get_project_dir
from .postprocesser import NoDetectedQuoteConventionException, PostprocessHandler, UnknownQuoteConventionException
from .usfm_utils import PARAGRAPH_TYPE_EMBEDS

LOGGER = logging.getLogger(__package__ + ".translate")
nltk.download("punkt")

CONFIDENCE_SCORES_SUFFIX = ".confidences.tsv"

# A group of multiple translations of a single sentence
TranslationGroup = List[str]

# A list representing a single draft (one translation of each input sentence)
TranslatedDraft = List[str]


# A wrapper around List[TranslationGroup] that allows upstream consumers to view a
# list of translation groups as a collection of discrete drafts
class DraftGroup:
    def __init__(self, translation_groups: List[TranslationGroup]):
        self.translation_groups = translation_groups
        self.num_drafts: int = len(self.translation_groups[0])

    def get_drafts(self) -> List[TranslatedDraft]:
        translated_draft_sentences = [[] for _ in range(self.num_drafts)]

        for translation_group in self.translation_groups:
            if len(translation_group) == 0:
                translation_group = self._createEmptyTranslationGroup()

            for draft_index in range(self.num_drafts):
                translated_draft_sentences[draft_index].append(translation_group[draft_index])

        return translated_draft_sentences

    def _createEmptyTranslationGroup(self):
        return ["" for _ in range(self.num_drafts)]


def generate_confidence_files(
    output: List[TranslationGroup],
    trg_file_path: Path,
    translate_step: bool = False,
    trg_prefix: str = "",
    produce_multiple_translations: bool = False,
    draft_index: int = 0,
    vrefs: Optional[List[VerseRef]] = None,
) -> None:
    if produce_multiple_translations:
        confidences_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}{CONFIDENCE_SCORES_SUFFIX}")
    else:
        confidences_path = trg_file_path.with_suffix(f"{trg_file_path.suffix}{CONFIDENCE_SCORES_SUFFIX}")
    sequence_confidences: List[float] = []
    ext = trg_file_path.suffix.lower()
    with confidences_path.open("w", encoding="utf-8", newline="\n") as confidences_file:
        if translate_step and ext in {".usfm", ".sfm"}:
            row1_col1_header = "VRef"
        else:
            row1_col1_header = "Sequence Number"
        confidences_file.write("\t".join([f"{row1_col1_header}"] + [f"Token {i}" for i in range(200)]) + "\n")
        confidences_file.write("\t".join(["Sequence Score"] + [f"Token Score {i}" for i in range(200)]) + "\n")
        for sentence_num, _ in enumerate(output):
            if output[sentence_num][0] is None:
                continue
            sequence_label = [str(sentence_num)]
            if translate_step:
                if ext in {".usfm", ".sfm"}:
                    sequence_label = [str(vrefs[sentence_num])]
                elif ext == ".txt":
                    sequence_confidences.append(exp(output[sentence_num][3][draft_index - 1]))
            confidences_file.write("\t".join(sequence_label + output[sentence_num][1][draft_index - 1]) + "\n")
            confidences_file.write(
                "\t".join(
                    [str(exp(output[sentence_num][3][draft_index - 1]))]
                    + [str(exp(token_score)) for token_score in output[sentence_num][2][draft_index - 1]]
                )
                + "\n"
            )
    if translate_step:
        if ext in {".usfm", ".sfm"}:
            chapter_confidences: DefaultDict[int, List[float]] = defaultdict(list)
            for sentence_num, vref in enumerate(vrefs):
                if not vref.is_verse or output[sentence_num][0] is None:
                    continue
                vref_confidence = exp(output[sentence_num][3][draft_index - 1])
                chapter_confidences[vref.chapter_num].append(vref_confidence)

            with confidences_path.with_suffix(".chapters.tsv").open(
                "w", encoding="utf-8", newline="\n"
            ) as chapter_confidences_file:
                chapter_confidences_file.write("Chapter\tConfidence\n")
                for chapter, confidences in chapter_confidences.items():
                    sequence_confidences += confidences
                    chapter_confidence = gmean(confidences)
                    chapter_confidences_file.write(f"{chapter}\t{chapter_confidence}\n")

            file_confidences_path = trg_file_path.parent / "confidences.books.tsv"
            row1_col1_header = "Book"
            if vrefs:
                col1_entry = vrefs[0].book
            else:
                col1_entry = trg_file_path.stem
        elif ext == ".txt":
            file_confidences_path = trg_file_path.parent / f"{trg_prefix}confidences.files.tsv"
            row1_col1_header = "File"
            col1_entry = trg_file_path.name
        else:
            raise ValueError(
                f"Invalid trg file extension {ext} when using --save-confidences in the translate step."
                f"Valid file extensions for --save-confidences are .usfm, .sfm, and .txt."
            )
        with file_confidences_path.open("a", encoding="utf-8", newline="\n") as file_confidences_file:
            if file_confidences_file.tell() == 0:
                file_confidences_file.write(f"{row1_col1_header}\tConfidence\n")
            file_confidences_file.write(f"{col1_entry}\t{gmean(sequence_confidences)}\n")


class Translator(ABC):
    @abstractmethod
    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        vrefs: Optional[Iterable[VerseRef]] = None,
    ) -> Iterable[TranslationGroup]:
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
    ) -> None:
        output = list(self.translate(load_corpus(src_file_path), src_iso, trg_iso, produce_multiple_translations))
        translations = [translation for translation, _, _, _ in output]
        draft_set = DraftGroup(translations)
        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            if produce_multiple_translations:
                trg_draft_file_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
            else:
                trg_draft_file_path = trg_file_path
            write_corpus(trg_draft_file_path, translated_draft)

            if save_confidences:
                generate_confidence_files(
                    output,
                    trg_file_path,
                    translate_step=True,
                    trg_prefix=trg_prefix,
                    produce_multiple_translations=produce_multiple_translations,
                    draft_index=draft_index,
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
            src_project,
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
        src_project: Optional[str] = None,
    ) -> None:
        # Create UsfmFileText object for source
        src_from_project = False
        if str(src_file_path).startswith(str(get_project_dir(""))):
            src_from_project = True
            src_settings = FileParatextProjectSettingsParser(src_file_path.parent).parse()
            src_file_text = UsfmFileText(
                src_settings.stylesheet,
                src_settings.encoding,
                src_settings.get_book_id(src_file_path.name),
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

            src_file_text = UsfmFileText("usfm.sty", "utf-8-sig", book_id, src_file_path, include_all_text=True)
        stylesheet = src_settings.stylesheet if src_from_project else UsfmStylesheet("usfm.sty")

        sentences = [re.sub(" +", " ", s.text.strip()) for s in src_file_text]
        vrefs = [s.ref for s in src_file_text]
        LOGGER.info(f"File {src_file_path} parsed correctly.")

        # Filter sentences
        for i in reversed(range(len(sentences))):
            marker = vrefs[i].path[-1].name if len(vrefs[i].path) > 0 else ""
            if (
                (len(chapters) > 0 and vrefs[i].chapter_num not in chapters)
                or marker in PARAGRAPH_TYPE_EMBEDS
                or stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT
            ):
                sentences.pop(i)
                vrefs.pop(i)
        empty_sents = []
        for i in reversed(range(len(sentences))):
            if len(sentences[i].strip()) == 0:
                sentences.pop(i)
                empty_sents.append((i, vrefs.pop(i)))

        output = list(self.translate(sentences, src_iso, trg_iso, produce_multiple_translations, vrefs))

        translations = [translation for translation, _, _, _ in output]

        # Add empty sentences back in
        # Prevents pre-existing text from showing up in the sections of translated text
        for idx, vref in reversed(empty_sents):
            sentences.insert(idx, "")
            translations.insert(idx, ["" for _ in range(len(translations[0]))])
            vrefs.insert(idx, vref)
            output.insert(idx, [None, None, None, None])

        text_behavior = (
            UpdateUsfmTextBehavior.PREFER_NEW if trg_project is not None else UpdateUsfmTextBehavior.STRIP_EXISTING
        )

        draft_set: DraftGroup = DraftGroup(translations)
        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            postprocess_handler.construct_rows(vrefs, sentences, translated_draft)

            for config in postprocess_handler.configs:

                # Compile draft remarks
                draft_src_str = f"project {src_file_text.project}" if src_from_project else f"file {src_file_path.name}"
                draft_remark = f"This draft of {vrefs[0].book} was machine translated on {date.today()} from {draft_src_str} using model {experiment_ckpt_str}. It should be reviewed and edited carefully."
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
                        id_text=vrefs[0].book,
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
                            config.create_denormalize_quotation_marks_postprocessor(training_corpus_pairs, src_project)
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
                        if not src_from_project
                        or src_from_project
                        and (src_settings.encoding == "utf-8-sig" or src_settings.encoding == "utf_8_sig")
                        else src_settings.encoding
                    ),
                ) as f:
                    f.write(usfm_out)

            if save_confidences:
                generate_confidence_files(
                    output,
                    trg_file_path,
                    translate_step=True,
                    produce_multiple_translations=produce_multiple_translations,
                    draft_index=draft_index,
                    vrefs=vrefs,
                )

    def translate_docx(
        self,
        src_file_path: Path,
        trg_file_path: Path,
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
    ) -> None:
        tokenizer: nltk.tokenize.PunktSentenceTokenizer
        try:
            src_lang = Lang(src_iso)
            tokenizer = nltk.data.load(f"tokenizers/punkt/{src_lang.name.lower()}.pickle")
        except:
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

        with src_file_path.open("rb") as file:
            doc = docx.Document(file)

        sentences: List[str] = []
        paras: List[int] = []

        for i in range(len(doc.paragraphs)):
            for sentence in tokenizer.tokenize(doc.paragraphs[i].text, "test"):
                sentences.append(sentence)
                paras.append(i)

        draft_set: DraftGroup = DraftGroup(
            list(self.translate(sentences, src_iso, trg_iso, produce_multiple_translations))
        )

        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            for para, group in groupby(zip(translated_draft, paras), key=lambda t: t[1]):
                text = " ".join(s[0] for s in group)
                doc.paragraphs[para].text = text

            if produce_multiple_translations:
                trg_draft_file_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
            else:
                trg_draft_file_path = trg_file_path

            with trg_draft_file_path.open("wb") as file:
                doc.save(file)
