from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

from machine.corpora import PlaceMarkersAlignmentInfo, PlaceMarkersUsfmUpdateBlockHandler, ScriptureRef
from machine.tokenization import LatinWordTokenizer
from machine.translation import WordAlignmentMatrix

from ..alignment.eflomal import to_word_alignment_matrix
from ..alignment.utils import compute_alignment_scores
from .corpus import load_corpus, write_corpus

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


def construct_place_markers_handler(
    refs: List[ScriptureRef], source: List[str], translation: List[str], aligner: str = "eflomal"
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
