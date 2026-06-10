import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence, Tuple

from machine.corpora import (
    PlaceMarkersAlignmentInfo,
    PlaceMarkersUsfmUpdateBlockHandler,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmRow,
    UpdateUsfmTextBehavior,
    UsfmStylesheet,
    UsfmToken,
    UsfmTokenizer,
    UsfmTokenType,
    UsfmUpdateBlockHandler,
    parse_usfm,
)
from machine.punctuation_analysis import (
    STANDARD_QUOTE_CONVENTIONS,
    Chapter,
    FileParatextProjectQuoteConventionDetector,
    QuotationMarkDenormalizationFirstPass,
    QuotationMarkDenormalizationUsfmUpdateBlockHandler,
    QuotationMarkUpdateSettings,
    QuotationMarkUpdateStrategy,
    QuoteConvention,
    QuoteConventionDetector,
)
from machine.tokenization import LatinWordTokenizer
from machine.translation import WordAlignmentMatrix

from ..alignment.eflomal import to_word_alignment_matrix
from ..alignment.utils import compute_alignment_scores
from ..nmt.corpora import CorpusPair
from .corpus import load_corpus, write_corpus
from .environment import SilNlpEnv

LOGGER = logging.getLogger((__package__ or "") + ".translate")

POSTPROCESS_DEFAULTS = {
    "paragraph_behavior": "end",  # Possible values: end, place, strip
    "include_style_markers": False,
    "include_embeds": False,
    "denormalize_quotation_marks": False,
    "target_quote_convention": "detect",
}
POSTPROCESS_SUFFIX_CHARS = {
    "paragraph_behavior": {"place": "p", "strip": "x"},
    "include_style_markers": "s",
    "include_embeds": "e",
    "denormalize_quotation_marks": "q",
}


class PlaceMarkersPostprocessor:
    _BEHAVIOR_DESCRIPTION_MAP = {
        UpdateUsfmMarkerBehavior.PRESERVE: " were preserved.",
        UpdateUsfmMarkerBehavior.STRIP: " were removed.",
    }

    def __init__(
        self,
        paragraph_behavior: UpdateUsfmMarkerBehavior,
        embed_behavior: UpdateUsfmMarkerBehavior,
        style_behavior: UpdateUsfmMarkerBehavior,
    ):
        self._paragraph_behavior = paragraph_behavior
        self._embed_behavior = embed_behavior
        self._style_behavior = style_behavior
        self._update_block_handlers = [PlaceMarkersUsfmUpdateBlockHandler()]

    def get_update_block_handlers(self) -> Sequence[UsfmUpdateBlockHandler]:
        return self._update_block_handlers

    def create_paragraph_remark(self) -> str:
        return self._create_remark_sentence_for_behavior(self._paragraph_behavior, ["paragraph break positions"])

    def _create_remark_sentence_for_behavior(self, behavior: UpdateUsfmMarkerBehavior, items: List[str]) -> str:
        return self._format_group_of_items_for_remark(items) + self._BEHAVIOR_DESCRIPTION_MAP[behavior]

    def _format_group_of_items_for_remark(self, items: List[str]) -> str:
        if len(items) == 1:
            return items[0].capitalize()
        elif len(items) == 2:
            return f"{items[0].capitalize()} and {items[1]}"
        return f"{items[0].capitalize()}, {', '.join(items[1:-1])}, and {items[-1]}"

    def postprocess_usfm(
        self,
        usfm: str,
        rows: List[UpdateUsfmRow],
        remarks: Optional[List[Tuple[int, str]]] = None,
        stylesheet: str | UsfmStylesheet = "usfm.sty",
    ) -> str:
        handler = UpdateUsfmParserHandler(
            rows=rows,
            text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
            paragraph_behavior=self._paragraph_behavior,
            embed_behavior=self._embed_behavior,
            style_behavior=self._style_behavior,
            update_block_handlers=self._update_block_handlers,
            remarks=remarks or [],
        )
        parse_usfm(usfm, handler, stylesheet)
        return handler.get_usfm(stylesheet)


class UnknownQuoteConventionException(Exception):
    def __init__(self, convention_name: str):
        super().__init__(f'"{convention_name}" is not a known quote convention.')
        self.convention_name = convention_name


class NoDetectedQuoteConventionException(Exception):
    def __init__(self, project_names: List[str]):
        if len(project_names) == 1:
            super().__init__(f'Could not detect quote convention for project "{project_names[0]}".')
        else:
            super().__init__(
                f'Could not detect quote convention for any of the following projects "{",".join(project_names)}".'
            )
        self.project_names = project_names


class DenormalizeQuotationMarksPostprocessor:
    _DENORMALIZED_CHAPTER_REMARK_SENTENCE = (
        "Quotation marks have been adjusted automatically to match the rest of the project."
    )
    _project_convention_cache: Dict[str, QuoteConvention] = {}

    def __init__(
        self,
        target_quote_convention_name: str | None,
        target_project_name: str | None = None,
        include_chapters: Optional[Dict[int, List[int]]] = None,
        environment: SilNlpEnv = SilNlpEnv.create_standard_environment(),
    ):
        self._environment = environment
        self._target_quote_convention = self._get_target_quote_convention(
            target_quote_convention_name, target_project_name, include_chapters
        )

    def _get_target_quote_convention(
        self,
        convention_name: str | None,
        project_name: str | None,
        include_chapters: Optional[Dict[int, List[int]]] = None,
    ) -> QuoteConvention:
        if convention_name is None or convention_name == "detect":
            if project_name is None:
                raise ValueError(
                    "The experiment's config.yml must exist and specify a target project name, "
                    "since an explicit target quote convention name was not provided."
                )
            return self._detect_quote_convention(project_name, include_chapters)
        return self._get_named_quote_convention(convention_name)

    def _get_named_quote_convention(self, convention_name: str) -> QuoteConvention:
        convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(convention_name)

        if convention is None:
            raise UnknownQuoteConventionException(convention_name)
        return convention

    def _detect_quote_convention(
        self, project_name: str, include_chapters: Optional[Dict[int, List[int]]] = None
    ) -> QuoteConvention:
        if project_name in self._project_convention_cache:
            return self._project_convention_cache[project_name]

        quote_convention_detector = QuoteConventionDetector()

        try:
            quote_convention_detector = FileParatextProjectQuoteConventionDetector(
                self._environment.get_paratext_project_dir(project_name)
            )
            quote_convention_analysis = quote_convention_detector.get_quote_convention_analysis(
                include_chapters=include_chapters
            )
        except ValueError as verr:
            raise NoDetectedQuoteConventionException([project_name]) from verr

        if quote_convention_analysis is None or quote_convention_analysis.best_quote_convention is None:
            raise NoDetectedQuoteConventionException([project_name])
        LOGGER.info(
            f'Detected quote convention for project "{project_name}" is '
            + f'"{quote_convention_analysis.best_quote_convention.name}" with score '
            + f"{quote_convention_analysis.best_quote_convention_score:.2f}."
        )
        self._project_convention_cache[project_name] = quote_convention_analysis.best_quote_convention

        return quote_convention_analysis.best_quote_convention

    def _create_update_block_handlers(
        self, chapter_strategies: List[QuotationMarkUpdateStrategy]
    ) -> List[UsfmUpdateBlockHandler]:
        return [
            QuotationMarkDenormalizationUsfmUpdateBlockHandler(
                self._target_quote_convention,
                QuotationMarkUpdateSettings(chapter_strategies=chapter_strategies),
            )
        ]

    def _get_best_chapter_strategies(self, usfm: str, stylesheet: UsfmStylesheet) -> List[QuotationMarkUpdateStrategy]:
        quotation_mark_update_first_pass = QuotationMarkDenormalizationFirstPass(self._target_quote_convention)

        parse_usfm(usfm, quotation_mark_update_first_pass, stylesheet)

        # sil-machine's get_chapters() currently mislabels chapter numbers, temp workaround until machine.py is fixed
        strategy_by_chapter: Dict[int, QuotationMarkUpdateStrategy] = {}
        for chapter, (_, strategy) in zip(
            quotation_mark_update_first_pass.get_chapters(),
            quotation_mark_update_first_pass.find_best_chapter_strategies(),
        ):
            chapter_num = self._get_chapter_number(chapter)
            if chapter_num is None:
                LOGGER.warning(
                    "Could not determine a chapter's number while denormalizing quotation marks; skipping it."
                )
                continue
            strategy_by_chapter[chapter_num] = strategy

        chapter_strategies = [QuotationMarkUpdateStrategy.APPLY_FULL] * max(strategy_by_chapter, default=0)
        for chapter_num, strategy in strategy_by_chapter.items():
            chapter_strategies[chapter_num - 1] = strategy
        return chapter_strategies

    @staticmethod
    def _get_chapter_number(chapter: Chapter) -> Optional[int]:
        for verse in chapter.verses:
            for text_segment in verse.text_segments:
                if text_segment.chapter is not None:
                    return text_segment.chapter
        return None

    def _create_chapter_remarks(self, chapter_strategies: List[QuotationMarkUpdateStrategy]) -> Dict[int, str]:
        return {
            chapter_num: self._DENORMALIZED_CHAPTER_REMARK_SENTENCE
            for chapter_num, strategy in enumerate(chapter_strategies, 1)
            if strategy != QuotationMarkUpdateStrategy.SKIP
        }

    @staticmethod
    def _append_sentences_to_last_rem(tokens: List[UsfmToken], chapter_sentences: Dict[int, str]) -> None:
        # Append each chapter's quotation sentence to the last \rem in that chapter (the draft remark, which is
        # always inserted after any pre-existing rems). The machine handler can't merge into an existing \rem, so
        # this is done at the token level. Chapters with no \rem (and book-level chapter 0) are left untouched.
        last_rem_index_by_chapter: Dict[int, int] = {}
        current_chapter = 0
        for index, token in enumerate(tokens):
            if token.type == UsfmTokenType.CHAPTER:
                data = str(token.data).strip() if token.data is not None else ""
                if data.isdigit():
                    current_chapter = int(data)
            elif token.type == UsfmTokenType.PARAGRAPH and token.marker == "rem":
                last_rem_index_by_chapter[current_chapter] = index

        # Apply deepest index first so inserting a TEXT token can't shift an earlier chapter's recorded index.
        for chapter_num in sorted(last_rem_index_by_chapter, reverse=True):
            if chapter_num == 0:
                continue
            sentence = chapter_sentences.get(chapter_num)
            if not sentence:
                continue
            next_index = last_rem_index_by_chapter[chapter_num] + 1
            if next_index < len(tokens) and tokens[next_index].type == UsfmTokenType.TEXT:
                existing_text = tokens[next_index].text or ""
                separator = "" if existing_text == "" or existing_text.endswith(" ") else " "
                tokens[next_index].text = existing_text + separator + sentence
            else:
                tokens.insert(next_index, UsfmToken(UsfmTokenType.TEXT, text=sentence))

    def postprocess_usfm(self, usfm: str, stylesheet: str | UsfmStylesheet = "usfm.sty") -> str:
        # The draft \rem is written by the draft-building step (translator's update_usfm, or the existing draft
        # file); this step only denormalizes the quotation marks and appends its note to that existing \rem.
        if isinstance(stylesheet, str):
            stylesheet = UsfmStylesheet(stylesheet)

        best_chapter_strategies = self._get_best_chapter_strategies(usfm, stylesheet)
        chapter_sentences = self._create_chapter_remarks(best_chapter_strategies)

        handler = UpdateUsfmParserHandler(
            update_block_handlers=self._create_update_block_handlers(best_chapter_strategies),
        )
        parse_usfm(usfm, handler, stylesheet)

        tokenizer = UsfmTokenizer(stylesheet)
        out_tokens = tokenizer.tokenize(handler.get_usfm(stylesheet))
        self._append_sentences_to_last_rem(out_tokens, chapter_sentences)
        return tokenizer.detokenize(out_tokens)


class PostprocessConfig:
    def __init__(self, config: dict = {}, environment: SilNlpEnv = SilNlpEnv.create_standard_environment()) -> None:
        self._config = {}
        self._environment = environment
        for option, default in POSTPROCESS_DEFAULTS.items():
            self._config[option] = config.get(option, default)

        # Backwards compatibility
        if config.get("include_paragraph_markers") or config.get("preserve_usfm_markers"):
            self._config["paragraph_behavior"] = "place"
        if config.get("preserve_usfm_markers"):
            self._config["include_style_markers"] = True
        if config.get("include_inline_elements"):
            self._config["include_embeds"] = True

        self.update_block_handlers: List[UsfmUpdateBlockHandler] = []
        self.rows: List[UpdateUsfmRow] = []

    def _get_usfm_marker_behavior(self, preserve: bool) -> UpdateUsfmMarkerBehavior:
        return UpdateUsfmMarkerBehavior.PRESERVE if preserve else UpdateUsfmMarkerBehavior.STRIP

    def get_paragraph_behavior(self) -> UpdateUsfmMarkerBehavior:
        return self._get_usfm_marker_behavior(self._config["paragraph_behavior"] != "strip")

    def get_style_behavior(self) -> UpdateUsfmMarkerBehavior:
        return self._get_usfm_marker_behavior(self._config["include_style_markers"])

    def get_embed_behavior(self) -> UpdateUsfmMarkerBehavior:
        return self._get_usfm_marker_behavior(self._config["include_embeds"])

    # NOTE: Each postprocessing configuration needs to have a unique suffix so files don't overwrite each other
    def get_postprocess_suffix(self) -> str:
        suffix = "_"
        for option, default in POSTPROCESS_DEFAULTS.items():
            if option in POSTPROCESS_SUFFIX_CHARS and self._config[option] != default:
                if isinstance(default, str):
                    suffix += POSTPROCESS_SUFFIX_CHARS[option][self._config[option]]
                else:
                    suffix += POSTPROCESS_SUFFIX_CHARS[option]

        return suffix if len(suffix) > 1 else ""

    def get_paragraph_marker_remark(self) -> Optional[str]:
        if self._config["paragraph_behavior"] not in ("place", "strip"):
            return None
        return self.create_place_markers_postprocessor().create_paragraph_remark()

    def is_base_config(self) -> bool:
        return self._config == POSTPROCESS_DEFAULTS

    def is_marker_placement_required(self) -> bool:
        return self._config["paragraph_behavior"] == "place" or self._config["include_style_markers"]

    def is_marker_processing_required(self) -> bool:
        return (
            self._config["paragraph_behavior"] != "end"
            or self._config["include_style_markers"]
            or self._config["include_embeds"]
        )

    def is_quotation_mark_denormalization_required(self) -> bool:
        return self._config["denormalize_quotation_marks"]

    def is_quote_convention_detection_required(self) -> bool:
        return self.is_quotation_mark_denormalization_required() and (
            self._config["target_quote_convention"] is None or self._config["target_quote_convention"] == "detect"
        )

    def create_place_markers_postprocessor(self) -> PlaceMarkersPostprocessor:
        return PlaceMarkersPostprocessor(
            paragraph_behavior=self.get_paragraph_behavior(),
            embed_behavior=self.get_embed_behavior(),
            style_behavior=self.get_style_behavior(),
        )

    def create_denormalize_quotation_marks_postprocessor(
        self, training_corpus_pairs: List[CorpusPair]
    ) -> DenormalizeQuotationMarksPostprocessor:
        training_project_info = self._get_training_project_info(
            training_corpus_pairs,
        )
        for training_target_project_name, include_chapters in training_project_info:

            try:
                return DenormalizeQuotationMarksPostprocessor(
                    self._config["target_quote_convention"],
                    training_target_project_name,
                    include_chapters,
                    self._environment,
                )
            except NoDetectedQuoteConventionException:
                LOGGER.warning("No quote convention was detected for project %s" % training_target_project_name)

        raise NoDetectedQuoteConventionException(
            [project_name for project_name, _ in training_project_info if project_name is not None]
        )

    def _get_training_project_info(
        self,
        training_corpus_pairs: List[CorpusPair],
    ) -> List[Tuple[Optional[str], Optional[Dict[int, List[int]]]]]:
        # Target project info is only needed for quote convention detection
        if self.is_quote_convention_detection_required():
            if len(training_corpus_pairs) > 1:
                LOGGER.warning(
                    "The experiment has multiple corpus pairs. "
                    "Quotation mark denormalization is unlikely to work correctly in this scenario."
                )
            if len(training_corpus_pairs) > 0 and len(training_corpus_pairs[0].src_files) > 1:
                LOGGER.warning(
                    "The experiment has multiple source projects. "
                    "Quotation mark denormalization is unlikely to work correctly in this scenario."
                )
            if len(training_corpus_pairs) > 0 and len(training_corpus_pairs[0].trg_files) > 1:
                LOGGER.warning(
                    "The experiment has multiple target projects. "
                    "Quotation mark denormalization is unlikely to work correctly in this scenario."
                )

            if len(training_corpus_pairs) > 0 and len(training_corpus_pairs[0].trg_files) > 0:
                return [
                    (corpus_pair.trg_files[0].project, corpus_pair.corpus_books)
                    for corpus_pair in training_corpus_pairs
                ]

        return [(None, None)]

    def __getitem__(self, key):
        return self._config[key]


class PostprocessHandler:
    def __init__(
        self,
        configs: Optional[List[PostprocessConfig]] = None,
        include_base: bool = True,
        environment: SilNlpEnv = SilNlpEnv.create_standard_environment(),
    ) -> None:
        if configs is None:
            configs = []

        self.configs = ([PostprocessConfig({}, environment)] if include_base else []) + configs

    # NOTE: Row metadata may need to be created/recreated at different times
    # For example, the marker placement metadata needs to be recreated for each new draft
    # because it uses text alignment, but other metadata may only need to be created once overall,
    # or once per source project. This may change what part of the process we want this function to be called at
    def construct_rows(self, refs: List[ScriptureRef], source: List[str], translation: List[str]) -> None:
        for config in self.configs:
            config.rows = [UpdateUsfmRow([ref], t, {}) for ref, t in zip(refs, translation)]

        self._construct_place_markers_metadata(source, translation)

    def _construct_place_markers_metadata(
        self, source: List[str], translation: List[str], aligner: str = "eflomal"
    ) -> None:
        pm_configs = [
            config
            for config in self.configs
            if config["paragraph_behavior"] == "place" or config["include_style_markers"]
        ]
        if len(pm_configs) == 0:
            return

        tokenizer = LatinWordTokenizer()
        alignments = self._get_alignment_matrices(source, translation, aligner)
        for i, (s, t, alignment) in enumerate(zip(source, translation, alignments)):
            source_tokens = list(tokenizer.tokenize(s))
            translation_tokens = list(tokenizer.tokenize(t))

            for config in pm_configs:
                row_metadata = config.rows[i].metadata
                if row_metadata is not None:
                    row_metadata["alignment_info"] = PlaceMarkersAlignmentInfo(
                        source_tokens=source_tokens,
                        translation_tokens=translation_tokens,
                        alignment=alignment,
                        paragraph_behavior=(
                            UpdateUsfmMarkerBehavior.PRESERVE
                            if config["paragraph_behavior"] == "place"
                            else UpdateUsfmMarkerBehavior.STRIP
                        ),
                        style_behavior=config.get_style_behavior(),
                    )

    def _get_alignment_matrices(
        self, src_sents: List[str], trg_sents: List[str], aligner: str = "eflomal"
    ) -> List[WordAlignmentMatrix]:
        with TemporaryDirectory() as td:
            align_path = Path(td, "sym-align.txt")
            write_corpus(Path(td, "src_align.txt"), src_sents)
            write_corpus(Path(td, "trg_align.txt"), trg_sents)
            compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), aligner, align_path)

            return [to_word_alignment_matrix(line) for line in load_corpus(align_path)]
