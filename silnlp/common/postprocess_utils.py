import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

import yaml
from machine.corpora import ScriptureRef, UsfmFileText, UsfmStylesheet, UsfmTextType
from machine.scripture import book_number_to_id, get_chapters
from machine.translation import WordAlignmentMatrix
from transformers.trainer_utils import get_last_checkpoint

from ..alignment.eflomal import to_word_alignment_matrix
from ..alignment.utils import compute_alignment_scores
from ..nmt.config import Config
from ..nmt.hugging_face_config import get_best_checkpoint
from .corpus import load_corpus, write_corpus
from .paratext import book_file_name_digits, get_book_path

LOGGER = logging.getLogger(__package__ + ".postprocess_utils")

# Marker "type" is as defined by the UsfmTokenType given to tokens by the UsfmTokenizer,
# which mostly aligns with a marker's StyleType in the USFM stylesheet
CHARACTER_TYPE_EMBEDS = ["fig", "fm", "jmp", "rq", "va", "vp", "xt", "xtSee", "xtSeeAlso"]
PARAGRAPH_TYPE_EMBEDS = ["lit", "r", "rem"]
NON_NOTE_TYPE_EMBEDS = CHARACTER_TYPE_EMBEDS + PARAGRAPH_TYPE_EMBEDS


def get_alignment_matrices(
    src_sents: List[str], trg_sents: List[str], aligner: str = "eflomal"
) -> List[WordAlignmentMatrix]:
    with TemporaryDirectory() as td:
        align_path = Path(td, "sym-align.txt")
        write_corpus(Path(td, "src_align.txt"), src_sents)
        write_corpus(Path(td, "trg_align.txt"), trg_sents)
        compute_alignment_scores(Path(td, "src_align.txt"), Path(td, "trg_align.txt"), aligner, align_path)

        return [to_word_alignment_matrix(line) for line in load_corpus(align_path)]


# NOTE: to be replaced by new machine.py remark functionality
def insert_draft_remarks(usfm: str, remarks: List[str]) -> str:
    lines = usfm.split("\n")
    remark_lines = [f"\\rem {r}" for r in remarks]
    return "\n".join(lines[:1] + remark_lines + lines[1:])


# Takes the path to a USFM file and the relevant info to parse it
# and returns the text of all non-embed sentences and their respective references,
# along with any remarks (\rem) that were inserted at the beginning of the file
def get_sentences(
    book_path: Path, stylesheet: UsfmStylesheet, encoding: str, book: str, chapters: List[int] = []
) -> Tuple[List[str], List[ScriptureRef], List[str]]:
    sents = []
    refs = []
    draft_remarks = []
    for sent in UsfmFileText(stylesheet, encoding, book, book_path, include_all_text=True):
        marker = sent.ref.path[-1].name if len(sent.ref.path) > 0 else ""
        if marker == "rem" and len(refs) == 0:  # TODO: \ide and \usfm lines could potentially come before the remark(s)
            draft_remarks.append(sent.text)
            continue
        if (
            marker in PARAGRAPH_TYPE_EMBEDS
            or stylesheet.get_tag(marker).text_type == UsfmTextType.NOTE_TEXT
            or (len(chapters) > 0 and sent.ref.chapter_num not in chapters)
        ):
            continue

        sents.append(re.sub(" +", " ", sent.text.strip()))
        refs.append(sent.ref)

    return sents, refs, draft_remarks


# Get the paths of all drafts that would be produced by an experiment's translate config and that exist
def get_draft_paths_from_exp(config: Config) -> Tuple[List[Path], List[Path]]:
    with (config.exp_dir / "translate_config.yml").open("r", encoding="utf-8") as file:
        translate_requests = yaml.safe_load(file).get("translate", [])

    src_paths = []
    draft_paths = []
    for translate_request in translate_requests:
        src_project = translate_request.get("src_project", next(iter(config.src_projects)))

        ckpt = translate_request.get("checkpoint", "last")
        if ckpt == "best":
            step_str = get_best_checkpoint(config.model_dir).name[11:]
        elif ckpt == "last":
            step_str = Path(get_last_checkpoint(config.model_dir)).name[11:]
        else:
            step_str = str(ckpt)

        book_nums = get_chapters(translate_request.get("books", [])).keys()
        for book_num in book_nums:
            book = book_number_to_id(book_num)

            src_path = get_book_path(src_project, book)
            draft_path = (
                config.exp_dir / "infer" / step_str / src_project / f"{book_file_name_digits(book_num)}{book}.SFM"
            )
            if draft_path.exists():
                src_paths.append(src_path)
                draft_paths.append(draft_path)
            elif draft_path.with_suffix(f".{1}{draft_path.suffix}").exists():  # multiple drafts
                for i in range(1, config.infer.get("num_drafts", 1) + 1):
                    src_paths.append(src_path)
                    draft_paths.append(draft_path.with_suffix(f".{i}{draft_path.suffix}"))
            else:
                LOGGER.warning(f"Draft not found: {draft_path}")

    return src_paths, draft_paths
