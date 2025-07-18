from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

from machine.corpora import (
    PlaceMarkersAlignmentInfo,
    PlaceMarkersUsfmUpdateBlockHandler,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmRow,
    UsfmUpdateBlockHandler,
)
from machine.tokenization import LatinWordTokenizer
from machine.translation import WordAlignmentMatrix

from ..alignment.eflomal import to_word_alignment_matrix
from ..alignment.utils import compute_alignment_scores
from .corpus import load_corpus, write_corpus

POSTPROCESS_DEFAULTS = {
    "paragraph_behavior": "end",  # Possible values: end, place, strip
    "include_style_markers": False,
    "include_embeds": False,
}
POSTPROCESS_SUFFIX_CHARS = [{"place": "p", "strip": "x"}, "s", "e"]


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

        if self._config["paragraph_behavior"] == "place" or self._config["include_style_markers"]:
            self.update_block_handlers.append(PlaceMarkersUsfmUpdateBlockHandler())

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
        for (option, default), char in zip(POSTPROCESS_DEFAULTS.items(), POSTPROCESS_SUFFIX_CHARS):
            if self._config[option] != default:
                if isinstance(default, str):
                    suffix += char[self._config[option]]
                else:
                    suffix += char

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
                config.rows[i].metadata["alignment_info"] = PlaceMarkersAlignmentInfo(
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
