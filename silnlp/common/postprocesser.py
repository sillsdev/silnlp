from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

from machine.corpora import (
    PlaceMarkersAlignmentInfo,
    PlaceMarkersUsfmUpdateBlockHandler,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UsfmUpdateBlockHandler,
)
from machine.tokenization import LatinWordTokenizer
from machine.translation import WordAlignmentMatrix

from ..alignment.eflomal import to_word_alignment_matrix
from ..alignment.utils import compute_alignment_scores
from .corpus import load_corpus, write_corpus
from .utils import merge_dict

POSTPROCESS_OPTIONS = {"include_paragraph_markers": False, "include_style_markers": False, "include_embeds": False}
POSTPROCESS_SUFFIX_CHARS = ["p", "s", "e"]


class PostprocessConfig:
    def __init__(self, config: Dict[str, Union[bool, str]]) -> None:
        self._config = merge_dict(dict(POSTPROCESS_OPTIONS), config)
        self.update_block_handlers: List[UsfmUpdateBlockHandler] = []

    def _get_usfm_marker_behavior(self, preserve: bool) -> UpdateUsfmMarkerBehavior:
        return UpdateUsfmMarkerBehavior.PRESERVE if preserve else UpdateUsfmMarkerBehavior.STRIP

    def get_paragraph_behavior(self) -> UpdateUsfmMarkerBehavior:
        return self._get_usfm_marker_behavior(self._config["include_paragraph_markers"])

    def get_style_behavior(self) -> UpdateUsfmMarkerBehavior:
        return self._get_usfm_marker_behavior(self._config["include_style_markers"])

    def get_embed_behavior(self) -> UpdateUsfmMarkerBehavior:
        return self._get_usfm_marker_behavior(self._config["include_embeds"])

    def get_postprocess_suffix(self) -> str:
        suffix = "_"
        for (option, default), char in zip(POSTPROCESS_OPTIONS.items(), POSTPROCESS_SUFFIX_CHARS):
            if self._config[option] != default:
                suffix += char

        return suffix if len(suffix) > 1 else ""

    def get_postprocess_remark(self) -> str:
        used = [option for (option, default) in POSTPROCESS_OPTIONS.items() if self._config[option] != default]
        return f"Post-processing options used: {' '.join(used)}" if len(used) > 0 else ""

    def __getitem__(self, key):
        return self._config[key]


class PostprocessHandler:
    def __init__(self, configs: List[Dict[str, Union[bool, str]]] = [], include_base: bool = True) -> None:
        self.configs = [PostprocessConfig(config) for config in ([{}] if include_base else []) + configs]

    # NOTE: Update block handlers may need to be created/recreated at different times
    # For example, the marker placement handler needs to be recreated for each new draft because it uses text alignment,
    # but other handlers may only need to be created once overall, or once per source project.
    # This may change what part of the process we want this function to be called at
    def create_update_block_handlers(self, refs: List[ScriptureRef], source: List[str], translation: List[str]) -> None:
        if any(config["include_paragraph_markers"] or config["include_style_markers"] for config in self.configs):
            place_markers_handler = self._construct_place_markers_handler(refs, source, translation)

        for config in self.configs:
            if config["include_paragraph_markers"] or config["include_style_markers"]:
                if len(config.update_block_handlers) == 0:
                    config.update_block_handlers.append(place_markers_handler)
                else:  # NOTE: this assumes a set order of update block handlers
                    config.update_block_handlers[0] = place_markers_handler

    def _construct_place_markers_handler(
        self, refs: List[ScriptureRef], source: List[str], translation: List[str], aligner: str = "eflomal"
    ) -> PlaceMarkersUsfmUpdateBlockHandler:
        align_info = []
        tokenizer = LatinWordTokenizer()
        alignments = self._get_alignment_matrices(source, translation, aligner)
        for ref, s, t, alignment in zip(refs, source, translation, alignments):
            align_info.append(
                PlaceMarkersAlignmentInfo(
                    refs=[str(ref)],
                    source_tokens=list(tokenizer.tokenize(s)),
                    translation_tokens=list(tokenizer.tokenize(t)),
                    alignment=alignment,
                )
            )
        return PlaceMarkersUsfmUpdateBlockHandler(align_info)

    def _get_alignment_matrices(
        self, src_sents: List[str], trg_sents: List[str], aligner: str = "eflomal"
    ) -> List[WordAlignmentMatrix]:
        with TemporaryDirectory() as td:
            align_path = Path(td, "sym-align.txt")
            write_corpus(Path(td, "src_align.txt"), src_sents)
            write_corpus(Path(td, "trg_align.txt"), trg_sents)
            compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), aligner, align_path)

            return [to_word_alignment_matrix(line) for line in load_corpus(align_path)]
