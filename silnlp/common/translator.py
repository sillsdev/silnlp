import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import date
from itertools import groupby
from pathlib import Path
from typing import Iterable, List, Optional

import docx
import nltk
from iso639 import Lang
from machine.corpora import (
    FileParatextProjectSettingsParser,
    FileParatextProjectTextUpdater,
    UpdateUsfmBehavior,
    UpdateUsfmParserHandler,
    UsfmFileText,
    UsfmParserState,
    UsfmStylesheet,
    UsfmStyleType,
    UsfmTextType,
    UsfmTokenizer,
    UsfmTokenType,
    parse_usfm,
)
from machine.scripture import VerseRef
from machine.translation import TranslationResult

from .corpus import load_corpus, write_corpus
from .paratext import get_book_path, get_iso, get_project_dir

LOGGER = logging.getLogger(__package__ + ".translate")
nltk.download("punkt")

NON_NOTE_INLINE_ELEMENTS = ["fm", "rq", "xtSeeAlso"]


class ParagraphUpdateUsfmParserHandler(UpdateUsfmParserHandler):
    def _collect_tokens(self, state: UsfmParserState) -> None:
        self._tokens.extend(self._new_tokens)
        self._new_tokens.clear()
        while self._token_index <= state.index + state.special_token_count:
            if (
                state.tokens[self._token_index].type == UsfmTokenType.PARAGRAPH
                and state.tokens[self._token_index].marker != "rem"
            ):
                num_text = 0
                rem_offset = 0
                for i in range(len(self._tokens) - 1, -1, -1):
                    if self._tokens[i].type == UsfmTokenType.TEXT:
                        num_text += 1
                    elif self._tokens[i].type == UsfmTokenType.PARAGRAPH and self._tokens[i].marker == "rem":
                        rem_offset += num_text + 1
                        num_text = 0
                    else:
                        break
                if num_text >= 2:
                    self._tokens.insert(-(rem_offset + num_text - 1), state.tokens[self._token_index])
                    self._token_index += 1
                    break  # should this be continue instead? is there just no difference bc only 1 paragraph marker is added at a time?
            self._tokens.append(state.tokens[self._token_index])
            self._token_index += 1


def insert_draft_remark(
    usfm: str,
    book: str,
    description: str,
    experiment_ckpt_str: str,
) -> str:
    remark = f"\\rem This draft of {book} was machine translated on {date.today()} from {description} using model {experiment_ckpt_str}. It should be reviewed and edited carefully.\n"

    lines = usfm.split("\n")
    insert_idx = (
        1
        + (len(lines) > 1 and (lines[1].startswith("\\ide") or lines[1].startswith("\\usfm")))
        + (len(lines) > 2 and (lines[2].startswith("\\ide") or lines[2].startswith("\\usfm")))
    )
    lines.insert(insert_idx, remark)
    return "\n".join(lines)


# A group of multiple translations of a single sentence
TranslationGroup = List[TranslationResult]

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
        draft_set: DraftGroup = DraftGroup(
            list(self.translate(load_corpus(src_file_path), src_iso, trg_iso, produce_multiple_translations))
        )
        for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
            if produce_multiple_translations:
                trg_draft_file_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
            else:
                trg_draft_file_path = trg_file_path
            write_corpus(trg_draft_file_path, translated_draft)

    def translate_book(
        self,
        src_project: str,
        book: str,
        output_path: Path,
        trg_iso: str,
        produce_multiple_translations: bool = False,
        chapters: List[int] = [],
        trg_project: Optional[str] = None,
        include_inline_elements: bool = False,
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
            get_iso(src_project),
            trg_iso,
            produce_multiple_translations,
            chapters,
            trg_project,
            include_inline_elements,
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
        include_inline_elements: bool = False,
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
                include_markers=True,
                include_all_text=True,
                project=src_settings.name,
            )
        else:
            src_file_text = UsfmFileText("usfm.sty", "utf-8-sig", "", src_file_path, include_all_text=True)

        sentences = [re.sub(" +", " ", s.text.strip()) for s in src_file_text]
        vrefs = [s.ref for s in src_file_text]
        LOGGER.info(f"File {src_file_path} parsed correctly.")

        # Filter sentences
        stylesheet = src_settings.stylesheet if src_from_project else UsfmStylesheet("usfm.sty")
        for i in reversed(range(len(sentences))):
            if len(chapters) > 0 and vrefs[i].chapter_num not in chapters:
                sentences.pop(i)
                vrefs.pop(i)
            elif not include_inline_elements:
                marker = vrefs[i].path[-1].name if len(vrefs[i].path) > 0 else ""
                if stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT or marker in NON_NOTE_INLINE_ELEMENTS:
                    sentences.pop(i)
                    vrefs.pop(i)
        # Set aside empty sentences
        empty_sents = []
        for i in reversed(range(len(sentences))):
            if len(sentences[i]) == 0:
                sentences.pop(i)
                empty_sents.append((i, vrefs.pop(i)))

        # NOTE: Parse sentences
        tokenizer = UsfmTokenizer(src_settings.stylesheet)
        sentence_toks = [tokenizer.tokenize(sent) for sent in sentences]

        to_delete = ["fig"]
        inline_markers = []  # NOTE: (sent_idx, start idx in text only sent, tok (inc. \s and spaces))
        text_only_src_sents = ["" for _ in sentence_toks]
        for i, toks in enumerate(sentence_toks):
            ignore_scope = None
            for tok in toks:
                if ignore_scope is not None:
                    if tok.type == UsfmTokenType.END and tok.marker[:-1] == ignore_scope.marker:
                        ignore_scope = None
                elif tok.type == UsfmTokenType.NOTE or (
                    tok.type == UsfmTokenType.CHARACTER and tok.marker in to_delete
                ):
                    ignore_scope = tok
                elif tok.type in [UsfmTokenType.PARAGRAPH, UsfmTokenType.CHARACTER, UsfmTokenType.END]:
                    inline_markers.append((i, len(text_only_src_sents[i]), tok.to_usfm()))
                elif tok.type == UsfmTokenType.TEXT:
                    text_only_src_sents[i] += tok.text
        print("\ninline markers")
        for m in inline_markers:
            print(m)

        # TranslationResult properties: translation, source/target tokens, confidences, sources, alignment, phrases
        translation_groups = self.translate(text_only_src_sents, src_iso, trg_iso, produce_multiple_translations, vrefs)
        translation_results = [r[0] for r in translation_groups]
        translations = [[tr.translation] for tr in translation_results]

        # NOTE: Map each token to a character range in the original strings
        # tokenizer: PreTrainedTokenizer = self._model._config.get_tokenizer()
        src_tok_ranges = []
        trg_tok_ranges = []
        for sent, tr in zip(text_only_src_sents, translation_results):
            # look at hf BatchEncoding methods (return type of tokenizer("text")) for alternatives if not working
            if (
                "".join(tr.source_tokens).replace("▁", " ")[1:]
                != sent  # orig: "".join(tr.source_tokens)[1:].replace("▁", " ")
                or "".join(tr.target_tokens).replace("▁", " ")[1:] != tr.translation
            ):
                print("bad")
                print(tr.source_tokens)
                print("".join(tr.source_tokens).replace("▁", " ")[1:])
                print(sent)
                print(tr.target_tokens)
                print("".join(tr.target_tokens).replace("▁", " ")[1:])
                print(tr.translation)
            src_sent_tok_ranges = [(0, len(tr.source_tokens[0]) - 1)]
            trg_sent_tok_ranges = [(0, len(tr.target_tokens[0]) - 1)]
            for tok in tr.source_tokens[1:]:
                if "▁" in tok[1:]:
                    print("bad2")
                src_sent_tok_ranges.append(
                    (src_sent_tok_ranges[-1][1] + (1 if tok[0] == "▁" else 0), src_sent_tok_ranges[-1][1] + len(tok))
                )
            src_tok_ranges.append(src_sent_tok_ranges)
            for tok in tr.target_tokens[1:]:
                if "▁" in tok[1:]:
                    print("bad2_trg")
                trg_sent_tok_ranges.append(
                    (trg_sent_tok_ranges[-1][1] + (1 if tok[0] == "▁" else 0), trg_sent_tok_ranges[-1][1] + len(tok))
                )
            trg_tok_ranges.append(trg_sent_tok_ranges)

        """NOTE: Match markers to their closest token idx"""
        toks_after_markers = []
        for marker in inline_markers:
            sent_idx, start_idx, _ = marker
            for i, tok_range in reversed(list(enumerate(src_tok_ranges[sent_idx]))):
                if tok_range[0] <= start_idx or i == 0:  # should this be < and then add i+1?
                    toks_after_markers.append(i)
                    break
        print("\nsrc toks after markers")
        print(toks_after_markers)
        for m, tok in zip(inline_markers, toks_after_markers):
            print(m)
            print(translation_results[m[0]].source_tokens[tok])
            try:
                print(translation_results[m[0]].source_tokens[tok - 1 : tok + 2])
            except:
                pass
            print(sentences[m[0]])

        """NOTE: Decide where to reinsert markers"""
        # seems like each target word is aligned to a single source word
        # so, a source word can be aligned to 0, 1, or more target words
        trg_toks_after_markers = []
        for idx, (sent_idx, _, _) in zip(toks_after_markers, inline_markers):
            offset = 0
            while True:
                try:
                    # if aligning with a token before the marker, get the last target token it aligns to
                    # if aligning with a token after the marker, get the first target token it aligns to
                    pos = -1 if offset < 0 else 0
                    trg_toks_after_markers.append(
                        list(translation_results[sent_idx].alignment.get_row_aligned_indices(idx + offset))[pos]
                    )
                    # TODO: could try adjusting trg idx to be "closer" to the marker (i.e. subtracting for 3 for offset 3)
                    break
                except:
                    # outward expanding search: offset = 0, -1, 1, -2, 2, ...
                    offset = -offset - (1 if offset >= 0 else 0)
        print("\ntrg toks after markers")
        print(trg_toks_after_markers)

        to_insert = [[] for _ in vrefs]
        # NOTE: Collect the markers to be inserted
        for mark, next_trg_tok in zip(inline_markers, trg_toks_after_markers):
            sent_idx, _, marker = mark
            trg_str_idx = trg_tok_ranges[sent_idx][next_trg_tok][0]

            # figure out the order of the markers in the sentence to handle ambiguity for directly adjacent markers
            insert_place = 0
            while insert_place < len(to_insert[sent_idx]) and to_insert[sent_idx][insert_place][0] <= trg_str_idx:
                insert_place += 1

            to_insert[sent_idx].insert(insert_place, (trg_str_idx, marker))
        print("\nto insert")
        for inserts, vref in zip(to_insert, vrefs):
            if len(inserts) > 0:
                print(vref, inserts)

        # NOTE: Construct rows to update the USFM file with
        # Create rows for each paragraph marker and insert character markers back into text
        rows = []
        for ref, tr, inserts in zip(vrefs, translation_results, to_insert):
            trg_sent = tr.translation
            if len(inserts) == 0:
                rows.append(([ref], trg_sent))
                continue

            row_texts = [trg_sent[: inserts[0][0]]]
            for i, (insert_idx, marker) in enumerate(inserts):
                is_char_marker = (
                    src_settings.stylesheet.get_tag(marker.strip(" \\+*")).style_type == UsfmStyleType.CHARACTER
                )
                if not is_char_marker:
                    row_texts.append("")

                row_text = (
                    (marker if is_char_marker else "")
                    + (" " if "*" in marker and insert_idx < len(trg_sent) and trg_sent[insert_idx].isalpha() else "")
                    + (trg_sent[insert_idx : inserts[i + 1][0]] if i + 1 < len(inserts) else trg_sent[insert_idx:])
                )
                # don't want a space before an end marker
                if i + 1 < len(inserts) and "*" in inserts[i + 1][1] and len(row_text) > 0 and row_text[-1] == " ":
                    row_text = row_text[:-1]
                row_texts[-1] += row_text

            for row_text in row_texts:
                rows.append(([ref], row_text))
        print("\nrows")
        for r in rows:
            print(r)

        """NOTE: Update USFM and write out"""
        with open(src_file_path, encoding="utf-8-sig") as f:
            usfm = f.read()
        handler = ParagraphUpdateUsfmParserHandler(rows, behavior=UpdateUsfmBehavior.PREFER_NEW)
        parse_usfm(usfm, handler, src_settings.stylesheet, src_settings.versification, preserve_whitespace=False)
        usfm_out = handler.get_usfm(src_settings.stylesheet)

        # Insert draft remark and write to output path
        description = f"project {src_file_text.project}" if src_from_project else f"file {src_file_path.name}"
        usfm_out = insert_draft_remark(usfm_out, vrefs[0].book, description, experiment_ckpt_str)
        with trg_file_path.open("w", encoding=src_settings.encoding) as f:
            f.write(usfm_out)

        # # Add empty sentences back in
        # for idx, vref in reversed(empty_sents):
        #     translations.insert(idx, [])
        #     vrefs.insert(idx, vref)

        # draft_set: DraftGroup = DraftGroup(translations)
        # for draft_index, translated_draft in enumerate(draft_set.get_drafts(), 1):
        #     rows = [([ref], translation) for ref, translation in zip(vrefs, translated_draft)]

        #     # Insert translation into the USFM structure of an existing project
        #     # If the target project is not the same as the translated file's original project,
        #     # no verses outside of the ones translated will be overwritten
        #     use_src_project = trg_project is None and src_from_project
        #     trg_format_project = src_file_path.parent.name if use_src_project else trg_project
        #     if trg_format_project is not None:
        #         dest_project_path = get_project_dir(trg_format_project)
        #         dest_updater = FileParatextProjectTextUpdater(dest_project_path)
        #         usfm_out = dest_updater.update_usfm(
        #             book_id=src_file_text.id,
        #             rows=rows,
        #             behavior=UpdateUsfmBehavior.STRIP_EXISTING if use_src_project else UpdateUsfmBehavior.PREFER_NEW,
        #         )

        #         if usfm_out is None:
        #             raise FileNotFoundError(f"Book {src_file_text.id} does not exist in target project {trg_project}")
        #     # Insert translation into the USFM structure of an individual file
        #     else:
        #         with open(src_file_path, encoding="utf-8-sig") as f:
        #             usfm = f.read()
        #         handler = UpdateUsfmParserHandler(
        #             rows=rows, id_text=vrefs[0].book, behavior=UpdateUsfmBehavior.STRIP_EXISTING
        #         )
        #         parse_usfm(usfm, handler)
        #         usfm_out = handler.get_usfm()

        #     # Insert draft remark and write to output path
        #     description = f"project {src_file_text.project}" if src_from_project else f"file {src_file_path.name}"
        #     usfm_out = insert_draft_remark(usfm_out, vrefs[0].book, description, experiment_ckpt_str)
        #     encoding = src_settings.encoding if src_from_project else "utf-8"

        #     if produce_multiple_translations:
        #         trg_draft_file_path = trg_file_path.with_suffix(f".{draft_index}{trg_file_path.suffix}")
        #     else:
        #         trg_draft_file_path = trg_file_path

        #     with trg_draft_file_path.open("w", encoding=encoding) as f:
        #         f.write(usfm_out)

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
