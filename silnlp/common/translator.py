import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import date
from itertools import groupby
from math import exp
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import docx
import nltk
from iso639 import Lang
from machine.corpora import (
    FileParatextProjectSettingsParser,
    FileParatextProjectTextUpdater,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
    UsfmFileText,
    UsfmStylesheet,
    UsfmTextType,
    parse_usfm,
)
from machine.scripture import VerseRef

from .corpus import load_corpus, write_corpus
from .paratext import get_book_path, get_iso, get_project_dir
from .usfm_preservation import PARAGRAPH_TYPE_EMBEDS, construct_place_markers_handler
from .utils import merge_dict

LOGGER = logging.getLogger(__package__ + ".translate")
nltk.download("punkt")


def insert_draft_remark(
    usfm: str,
    book: str,
    description: str,
    experiment_ckpt_str: str,
) -> str:
    remark = f"\\rem This draft of {book} was machine translated on {date.today()} from {description} using model {experiment_ckpt_str}. It should be reviewed and edited carefully."

    lines = usfm.split("\n")
    insert_idx = (
        1
        + (len(lines) > 1 and (lines[1].startswith("\\ide") or lines[1].startswith("\\usfm")))
        + (len(lines) > 2 and (lines[2].startswith("\\ide") or lines[2].startswith("\\usfm")))
    )
    lines.insert(insert_idx, remark)
    return "\n".join(lines)


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
    ) -> None:
        output = list(self.translate(load_corpus(src_file_path), src_iso, trg_iso, produce_multiple_translations))
        translations = [translation for translation, _, _, _ in output]
        draft_set = DraftGroup(translations)
        confidence_scores_suffix = ".confidences.tsv"
        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            if produce_multiple_translations:
                trg_draft_file_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
                confidences_path = trg_file_path.with_suffix(
                    f".{draft_index}{trg_file_path.suffix}{confidence_scores_suffix}"
                )
            else:
                trg_draft_file_path = trg_file_path
                confidences_path = trg_file_path.with_suffix(f"{trg_file_path.suffix}{confidence_scores_suffix}")
            write_corpus(trg_draft_file_path, translated_draft)
            with confidences_path.open("w", encoding="utf-8", newline="\n") as confidences_file:
                confidences_file.write("\t".join(["Sequence Number"] + [f"Token {i}" for i in range(200)]) + "\n")
                confidences_file.write("\t".join(["Sequence Score"] + [f"Token Score {i}" for i in range(200)]) + "\n")
                for sentence_num, _ in enumerate(output):
                    confidences_file.write(
                        "\t".join([str(sentence_num)] + output[sentence_num][1][draft_index - 1]) + "\n"
                    )
                    confidences_file.write(
                        "\t".join(
                            [str(exp(output[sentence_num][3][draft_index - 1]))]
                            + [str(exp(token_score)) for token_score in output[sentence_num][2][draft_index - 1]]
                        )
                        + "\n"
                    )

    def translate_book(
        self,
        src_project: str,
        book: str,
        output_path: Path,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        chapters: List[int] = [],
        trg_project: Optional[str] = None,
        postprocess_configs: List[Dict[str, bool]] = [],
        experiment_ckpt_str: str = "",
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
            chapters,
            trg_project,
            postprocess_configs,
            experiment_ckpt_str,
        )

    def translate_usfm(
        self,
        src_file_path: Path,
        trg_file_path: Path,
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        chapters: List[int] = [],
        trg_project: Optional[str] = None,
        postprocess_configs: List[Dict[str, bool]] = [],
        experiment_ckpt_str: str = "",
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
            src_file_text = UsfmFileText("usfm.sty", "utf-8-sig", "", src_file_path, include_all_text=True)
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

        # Prepare configs: add base draft and default value
        postprocess_configs = [merge_dict(defaultdict(lambda: False), ppc) for ppc in [{}] + postprocess_configs]

        draft_set: DraftGroup = DraftGroup(translations)
        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            rows = [([ref], translation) for ref, translation in zip(vrefs, translated_draft)]

            if any(ppc["include_paragraph_markers"] or ppc["include_style_markers"] for ppc in postprocess_configs):
                place_markers_handler = construct_place_markers_handler(vrefs, sentences, translated_draft)

            for postprocess_config in postprocess_configs:
                # Update behaviors
                text_behavior = (
                    UpdateUsfmTextBehavior.PREFER_NEW
                    if trg_project is not None
                    else UpdateUsfmTextBehavior.STRIP_EXISTING
                )
                paragraph_behavior = (
                    UpdateUsfmMarkerBehavior.PRESERVE
                    if postprocess_config["include_paragraph_markers"]
                    else UpdateUsfmMarkerBehavior.STRIP
                )
                style_behavior = (
                    UpdateUsfmMarkerBehavior.PRESERVE
                    if postprocess_config["include_style_markers"]
                    else UpdateUsfmMarkerBehavior.STRIP
                )
                embed_behavior = (
                    UpdateUsfmMarkerBehavior.PRESERVE
                    if postprocess_config["include_embeds"]
                    else UpdateUsfmMarkerBehavior.STRIP
                )
                marker_placement_suffix = (
                    "_"
                    + ("p" if postprocess_config["include_paragraph_markers"] else "")
                    + ("s" if postprocess_config["include_style_markers"] else "")
                    + ("e" if postprocess_config["include_embeds"] else "")
                )
                marker_placement_suffix = "" if len(marker_placement_suffix) == 1 else marker_placement_suffix

                update_block_handlers = []
                if postprocess_config["include_paragraph_markers"] or postprocess_config["include_style_markers"]:
                    update_block_handlers.append(place_markers_handler)

                # Insert translation into the USFM structure of an existing project
                # If the target project is not the same as the translated file's original project,
                # no verses outside of the ones translated will be overwritten
                if trg_project is not None or src_from_project:
                    dest_updater = FileParatextProjectTextUpdater(
                        get_project_dir(trg_project if trg_project is not None else src_file_path.parent.name)
                    )
                    usfm_out = dest_updater.update_usfm(
                        book_id=src_file_text.id,
                        rows=rows,
                        text_behavior=text_behavior,
                        paragraph_behavior=paragraph_behavior,
                        embed_behavior=embed_behavior,
                        style_behavior=style_behavior,
                        update_block_handlers=update_block_handlers,
                    )

                    if usfm_out is None:
                        raise FileNotFoundError(
                            f"Book {src_file_text.id} does not exist in target project {trg_project}"
                        )
                else:  # Slightly more manual version for updating an individual file
                    with open(src_file_path, encoding="utf-8-sig") as f:
                        usfm = f.read()
                    handler = UpdateUsfmParserHandler(
                        rows=rows,
                        id_text=vrefs[0].book,
                        text_behavior=text_behavior,
                        paragraph_behavior=paragraph_behavior,
                        embed_behavior=embed_behavior,
                        style_behavior=style_behavior,
                        update_block_handlers=update_block_handlers,
                    )
                    parse_usfm(usfm, handler)
                    usfm_out = handler.get_usfm()

                # Insert draft remark and write to output path
                description = f"project {src_file_text.project}" if src_from_project else f"file {src_file_path.name}"
                usfm_out = insert_draft_remark(usfm_out, vrefs[0].book, description, experiment_ckpt_str)
                trg_draft_file_path = trg_file_path.with_stem(trg_file_path.stem + marker_placement_suffix)
                if produce_multiple_translations:
                    trg_draft_file_path = trg_draft_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
                with trg_draft_file_path.open(
                    "w", encoding=src_settings.encoding if src_from_project else "utf-8"
                ) as f:
                    f.write(usfm_out)

            confidence_scores_suffix = ".confidences.tsv"
            if produce_multiple_translations:
                confidences_path = trg_file_path.with_suffix(
                    f".{draft_index}{trg_file_path.suffix}{confidence_scores_suffix}"
                )
            else:
                confidences_path = trg_file_path.with_suffix(f"{trg_file_path.suffix}{confidence_scores_suffix}")
            with confidences_path.open("w", encoding="utf-8", newline="\n") as confidences_file:
                confidences_file.write("\t".join(["VRef"] + [f"Token {i}" for i in range(200)]) + "\n")
                confidences_file.write("\t".join(["Sequence Score"] + [f"Token Score {i}" for i in range(200)]) + "\n")
                for sentence_num, _ in enumerate(output):
                    if output[sentence_num][0] is None:
                        continue
                    confidences_file.write(
                        "\t".join([str(vrefs[sentence_num])] + output[sentence_num][1][draft_index - 1]) + "\n"
                    )
                    confidences_file.write(
                        "\t".join(
                            [str(exp(output[sentence_num][3][draft_index - 1]))]
                            + [str(exp(token_score)) for token_score in output[sentence_num][2][draft_index - 1]]
                        )
                        + "\n"
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
