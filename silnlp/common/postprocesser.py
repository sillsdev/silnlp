import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

from machine.corpora import (
    PlaceMarkersAlignmentInfo,
    PlaceMarkersUsfmUpdateBlockHandler,
    QuotationMarkDenormalizationFirstPass,
    QuotationMarkDenormalizationUsfmUpdateBlockHandler,
    QuotationMarkUpdateSettings,
    QuotationMarkUpdateStrategy,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmRow,
    UpdateUsfmTextBehavior,
    UsfmUpdateBlockHandler,
    parse_usfm,
)
from machine.punctuation_analysis import (
    STANDARD_QUOTE_CONVENTIONS,
    QuoteConvention,
    QuoteConventionAnalysis,
    QuoteConventionDetector,
)
from machine.scripture import get_chapters
from machine.tokenization import LatinWordTokenizer
from machine.translation import WordAlignmentMatrix

from silnlp.common.paratext import parse_project
from silnlp.nmt.corpora import CorpusPair

from ..alignment.eflomal import to_word_alignment_matrix
from ..alignment.utils import compute_alignment_scores
from .corpus import load_corpus, write_corpus

LOGGER = logging.getLogger(__package__ + ".translate")

POSTPROCESS_DEFAULTS = {
    "paragraph_behavior": "end",  # Possible values: end, place, strip
    "include_style_markers": False,
    "include_embeds": False,
    "denormalize_quotation_marks": False,
    "source_quote_convention": "standard_english",
    "target_quote_convention": "standard_english",
}
POSTPROCESS_SUFFIX_CHARS = {
    "paragraph_behavior": {"place": "p", "strip": "x"},
    "include_style_markers": "s",
    "include_embeds": "e",
    "denormalize_quotation_marks": "q",
}


class PlaceMarkersPostprocessor:
    _BEHAVIOR_DESCRIPTION_MAP = {
        UpdateUsfmMarkerBehavior.PRESERVE: " have positions preserved.",
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

    def _create_remark(self) -> str:
        behavior_map: Dict[UpdateUsfmMarkerBehavior, List[str]] = {
            UpdateUsfmMarkerBehavior.PRESERVE: [],
            UpdateUsfmMarkerBehavior.STRIP: [],
        }
        behavior_map[self._paragraph_behavior].append("paragraph markers")
        behavior_map[self._embed_behavior].append("embed markers")
        behavior_map[self._style_behavior].append("style markers")

        remark_sentences = [
            self._create_remark_sentence_for_behavior(behavior, items)
            for behavior, items in behavior_map.items()
            if len(items) > 0
        ]
        return " ".join(remark_sentences)

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
        remarks: List[str] = [],
    ) -> str:
        handler = UpdateUsfmParserHandler(
            rows=rows,
            text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
            paragraph_behavior=self._paragraph_behavior,
            embed_behavior=self._embed_behavior,
            style_behavior=self._style_behavior,
            update_block_handlers=self._update_block_handlers,
            remarks=remarks + [self._create_remark()],
        )
        parse_usfm(usfm, handler)
        return handler.get_usfm()


class UnknownQuoteConventionException(Exception):
    def __init__(self, convention_name: str):
        super().__init__(
            f'"{convention_name}" is not a known quote convention. Skipping quotation mark denormalization.'
        )
        self.convention_name = convention_name


class NoDetectedQuoteConventionException(Exception):
    def __init__(self, project_name: str):
        super().__init__(
            f'Could not detect quote convention for project "{project_name}". Skipping quotation mark denormalization.'
        )
        self.project_name = project_name


class DenormalizeQuotationMarksPostprocessor:
    _REMARK_SENTENCE = (
        "Quotation marks in the following chapters have been automatically denormalized after translation: "
    )

    def __init__(
        self,
        source_quote_convention_name: str | None,
        target_quote_convention_name: str | None,
        source_project_name: str | None = None,
        target_project_name: str | None = None,
        selected_training_books: Dict[int, List[int]] = {},
    ):
        self._source_quote_convention = self._get_source_quote_convention(
            source_quote_convention_name, source_project_name, selected_training_books
        )
        self._target_quote_convention = self._get_target_quote_convention(
            target_quote_convention_name, target_project_name, selected_training_books
        )

    def _get_source_quote_convention(
        self, convention_name: str | None, project_name: str | None, selected_training_books: Dict[int, List[int]] = {}
    ) -> QuoteConvention:
        if convention_name is None or convention_name == "detect":
            if project_name is None:
                raise ValueError(
                    "The experiment's translate_config.yml must exist and specify a source project name, since an explicit source quote convention name was not provided."
                )
            if selected_training_books is None:
                raise ValueError(
                    "The experiment's config.yml must exist and specify selected training books, since an explicit source quote convention name was not provided."
                )
            return self._detect_quote_convention(project_name, selected_training_books)
        return self._get_named_quote_convention(convention_name)

    def _get_target_quote_convention(
        self, convention_name: str | None, project_name: str | None, selected_training_books: Dict[int, List[int]] = {}
    ) -> QuoteConvention:
        if convention_name is None or convention_name == "detect":
            if project_name is None:
                raise ValueError(
                    "The experiment's config.yml must exist and specify a target project name, since an explicit target quote convention name was not provided."
                )
            if selected_training_books is None:
                raise ValueError(
                    "The experiment's config.yml must exist and specify selected training books, since an explicit target quote convention name was not provided."
                )
            return self._detect_quote_convention(project_name, selected_training_books)
        return self._get_named_quote_convention(convention_name)

    def _get_named_quote_convention(self, convention_name: str) -> QuoteConvention:
        convention = STANDARD_QUOTE_CONVENTIONS.get_quote_convention_by_name(convention_name)

        if convention is None:
            raise UnknownQuoteConventionException(convention_name)
        return convention

    def _detect_quote_convention(
        self, project_name: str, selected_training_books: Dict[int, List[int]] = {}
    ) -> QuoteConvention:
        quote_convention_detector = QuoteConventionDetector()

        parse_project(project_name, selected_training_books.keys(), quote_convention_detector)

        quote_convention_analysis: QuoteConventionAnalysis | None = quote_convention_detector.detect_quote_convention()
        if quote_convention_analysis is None:
            raise NoDetectedQuoteConventionException(project_name)
        return quote_convention_analysis.best_quote_convention

    def _create_update_block_handlers(
        self, chapter_strategies: List[QuotationMarkUpdateStrategy]
    ) -> List[UsfmUpdateBlockHandler]:
        return [
            QuotationMarkDenormalizationUsfmUpdateBlockHandler(
                self._source_quote_convention,
                self._target_quote_convention,
                QuotationMarkUpdateSettings(chapter_strategies=chapter_strategies),
            )
        ]

    def _get_best_chapter_strategies(self, usfm: str) -> List[QuotationMarkUpdateStrategy]:
        quotation_mark_update_first_pass = QuotationMarkDenormalizationFirstPass(
            self._source_quote_convention, self._target_quote_convention
        )

        parse_usfm(usfm, quotation_mark_update_first_pass)
        return quotation_mark_update_first_pass.find_best_chapter_strategies()

    def _create_remark(self, best_chapter_strategies: List[QuotationMarkUpdateStrategy]) -> str:
        return (
            self._REMARK_SENTENCE
            + ", ".join(
                [
                    str(chapter_num)
                    for chapter_num, strategy in enumerate(best_chapter_strategies, 1)
                    if strategy != QuotationMarkUpdateStrategy.SKIP
                ]
            )
            + "."
        )

    def postprocess_usfm(
        self,
        usfm: str,
    ) -> str:
        best_chapter_strategies = self._get_best_chapter_strategies(usfm)
        handler = UpdateUsfmParserHandler(
            update_block_handlers=self._create_update_block_handlers(best_chapter_strategies),
            remarks=[self._create_remark(best_chapter_strategies)],
        )
        parse_usfm(usfm, handler)
        return handler.get_usfm()


class PostprocessConfig:
    def __init__(self, config: dict = {}) -> None:
        self._config = {}
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

    def get_postprocess_remark(self) -> Optional[str]:
        used = []
        for option, default in POSTPROCESS_DEFAULTS.items():
            if self._config[option] != default:
                used.append(option)
                if isinstance(default, str):
                    used[-1] += f":{self._config[option]}"

        return f"Post-processing options used: {' '.join(used)}" if len(used) > 0 else None

    def is_base_config(self) -> bool:
        return self._config == POSTPROCESS_DEFAULTS

    def is_marker_placement_required(self) -> bool:
        return self._config["paragraph_behavior"] == "place" or self._config["include_style_markers"]

    def is_quotation_mark_denormalization_required(self) -> bool:
        return self._config["denormalize_quotation_marks"]

    def is_quote_convention_detection_required(self) -> bool:
        return self.is_quotation_mark_denormalization_required() and (
            self._config["source_quote_convention"] is None
            or self._config["source_quote_convention"] == "detect"
            or self._config["target_quote_convention"] is None
            or self._config["target_quote_convention"] == "detect"
        )

    def create_place_markers_postprocessor(self) -> PlaceMarkersPostprocessor:
        return PlaceMarkersPostprocessor(
            paragraph_behavior=self.get_paragraph_behavior(),
            embed_behavior=self.get_embed_behavior(),
            style_behavior=self.get_style_behavior(),
        )

    def create_denormalize_quotation_marks_postprocessor(
        self,
        training_corpus_pairs: List[CorpusPair],
    ) -> DenormalizeQuotationMarksPostprocessor:
        _, training_target_project_name, selected_training_books = self._get_experiment_training_info(
            training_corpus_pairs,
        )
        translation_source_project_name = self._config.get("src_project")

        return DenormalizeQuotationMarksPostprocessor(
            self._config["source_quote_convention"],
            self._config["target_quote_convention"],
            translation_source_project_name,
            training_target_project_name,
            selected_training_books,
        )

    def _get_experiment_training_info(
        self,
        training_corpus_pairs: List[CorpusPair],
    ) -> Tuple[Optional[str], Optional[str], Dict[int, List[int]]]:
        # Target project info is only needed for quote convention detection
        if self.is_quote_convention_detection_required():
            if len(training_corpus_pairs) > 1:
                LOGGER.warning(
                    "The experiment has multiple corpus pairs. Quotation mark denormalization is unlikely to work correctly in this scenario."
                )
            if len(training_corpus_pairs) > 0 and len(training_corpus_pairs[0].src_files) > 1:
                LOGGER.warning(
                    "The experiment has multiple source projects. Quotation mark denormalization is unlikely to work correctly in this scenario."
                )
            if len(training_corpus_pairs) > 0 and len(training_corpus_pairs[0].trg_files) > 1:
                LOGGER.warning(
                    "The experiment has multiple target projects. Quotation mark denormalization is unlikely to work correctly in this scenario."
                )

            source_project_name = (
                training_corpus_pairs[0].src_files[0].project
                if len(training_corpus_pairs) > 0 and len(training_corpus_pairs[0].src_files) > 0
                else None
            )
            target_project_name = (
                training_corpus_pairs[0].trg_files[0].project
                if len(training_corpus_pairs) > 0 and len(training_corpus_pairs[0].trg_files) > 0
                else None
            )
            selected_training_books = training_corpus_pairs[0].corpus_books if len(training_corpus_pairs) > 0 else {}

            return source_project_name, target_project_name, selected_training_books

        return None, None, {}

    def __getitem__(self, key):
        return self._config[key]


class PostprocessHandler:
    def __init__(self, configs: List[PostprocessConfig] = [], include_base: bool = True) -> None:
        self.configs = ([PostprocessConfig()] if include_base else []) + configs

    # NOTE: Row metadata may need to be created/recreated at different times
    # For example, the marker placement metadata needs to be recreated for each new draft because it uses text alignment,
    # but other metadata may only need to be created once overall, or once per source project.
    # This may change what part of the process we want this function to be called at
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
