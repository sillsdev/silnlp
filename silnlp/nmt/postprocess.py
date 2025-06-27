import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from machine.corpora import (
    FileParatextProjectSettingsParser,
    PlaceMarkersAlignmentInfo,
    PlaceMarkersUsfmUpdateBlockHandler,
    ScriptureRef,
    UpdateUsfmMarkerBehavior,
    UpdateUsfmParserHandler,
    UpdateUsfmTextBehavior,
    UsfmStylesheet,
    UsfmUpdateBlockHandler,
    parse_usfm,
)
from machine.tokenization import LatinWordTokenizer

from ..common.paratext import get_project_dir
from ..common.postprocess_utils import (
    get_alignment_matrices,
    get_draft_paths_from_exp,
    get_sentences,
    insert_draft_remarks,
)
from ..common.utils import get_git_revision_hash, merge_dict
from .clearml_connection import SILClearML
from .config_utils import load_config

LOGGER = logging.getLogger(__package__ + ".postprocess")

POSTPROCESS_OPTIONS = {"include_paragraph_markers": False, "include_style_markers": False, "include_embeds": False}
POSTPROCESS_SUFFIX_CHARS = ["p", "s", "e"]


class PostprocessConfig:
    update_block_handlers: List[UsfmUpdateBlockHandler] = []

    def __init__(self, config: Dict[str, Union[bool, str]]) -> None:
        # TODO: need to make a copy of the default dict?
        self._config = merge_dict(POSTPROCESS_OPTIONS, config)

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
        for option, char in zip(POSTPROCESS_OPTIONS, POSTPROCESS_SUFFIX_CHARS):
            if self._config[option]:
                suffix += char

        return suffix if len(suffix) > 1 else ""

    def get_postprocess_remark(self) -> str:
        return f"Post-processing options used: {' '.join(opt for opt in POSTPROCESS_OPTIONS if self._config[opt])}"


class PostprocessHandler:
    # TODO: check if one of the configs is already all default?
    def __init__(self, configs: List[Dict[str, Union[bool, str]]] = [], include_base: bool = True) -> None:
        self.configs = [PostprocessConfig(config) for config in ([{}] if include_base else []) + configs]

    # NOTE: Update block handlers may need to be created/recreated at different times
    # For example, the marker placement handler needs to be recreated for each new draft because it uses text alignment,
    # but other handlers may only need to be created once overall, or once per source project.
    # This may change what part of the process we want this function to be called at
    def create_update_block_handlers(self, refs: List[ScriptureRef], source: List[str], translation: List[str]) -> None:
        # USFM marker placement handler needs to be recreated for each draft
        if any(config["include_paragraph_markers"] or config["include_style_markers"] for config in self.configs):
            place_markers_handler = self._construct_place_markers_handler(refs, source, translation)

        for config in self.configs:
            # TODO: make sure the configs are changing
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
        alignments = get_alignment_matrices(source, translation, aligner)
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


def postprocess_draft(
    src_path: Path,
    draft_path: Path,
    postprocess_handler: PostprocessHandler,
    book: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> None:
    if str(src_path).startswith(str(get_project_dir(""))):
        settings = FileParatextProjectSettingsParser(src_path.parent).parse()
        stylesheet = settings.stylesheet
        encoding = settings.encoding
        book = settings.get_book_id(src_path.name)
    else:
        stylesheet = UsfmStylesheet("usfm.sty")
        encoding = "utf-8-sig"

    src_sents, src_refs, _ = get_sentences(src_path, stylesheet, encoding, book)
    draft_sents, draft_refs, draft_remarks = get_sentences(draft_path, stylesheet, encoding, book)

    # Verify reference parity
    if len(src_refs) != len(draft_refs):
        LOGGER.warning(f"Can't process {src_path} and {draft_path}: Unequal number of verses/references")
        return
    for src_ref, draft_ref in zip(src_refs, draft_refs):
        if src_ref.to_relaxed() != draft_ref.to_relaxed():
            LOGGER.warning(
                f"Can't process {src_path} and {draft_path}: Mismatched ref, {src_ref} != {draft_ref}. Files must have the exact same USFM structure"
            )
            return

    postprocess_handler.create_update_block_handlers(src_refs, src_sents, draft_sents)

    with src_path.open(encoding=encoding) as f:
        usfm = f.read()
    rows = [([ref], sent) for ref, sent in zip(src_refs, draft_sents)]

    for config in postprocess_handler.configs:
        handler = UpdateUsfmParserHandler(
            rows=rows,
            id_text=book,
            text_behavior=UpdateUsfmTextBehavior.STRIP_EXISTING,
            paragraph_behavior=config.get_paragraph_behavior(),
            embed_behavior=config.get_embed_behavior(),
            style_behavior=config.get_style_behavior(),
            update_block_handlers=config.update_block_handlers,
        )
        parse_usfm(usfm, handler)
        usfm_out = handler.get_usfm()

        usfm_out = insert_draft_remarks(usfm_out, draft_remarks + [config.get_postprocess_remark()])

        if not out_dir:
            out_dir = draft_path.parent
        out_path = out_dir / f"{draft_path.stem}{config.get_postprocess_suffix()}{draft_path.suffix}"
        with out_path.open("w", encoding="utf-8" if encoding == "utf-8-sig" else encoding) as f:
            f.write(usfm_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess the drafts created by an NMT model")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    args = parser.parse_args()

    get_git_revision_hash()

    if args.clearml_queue is not None:
        if "cpu" not in args.clearml_queue:
            raise ValueError("Running this script on a GPU queue will not speed it up. Please only use CPU queues.")
        clearml = SILClearML(args.experiment, args.clearml_queue)
        config = clearml.config
    else:
        config = load_config(args.experiment.replace("\\", "/"))
    config.set_seed()

    src_paths, draft_paths = get_draft_paths_from_exp(config)
    with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
        postprocess_configs = yaml.safe_load(file).get("postprocess", [])

    postprocess_handler = PostprocessHandler(postprocess_configs)

    for src_path, draft_path in zip(src_paths, draft_paths):
        postprocess_draft(src_path, draft_path, postprocess_handler)


if __name__ == "__main__":
    main()
