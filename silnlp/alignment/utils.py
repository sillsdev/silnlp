import logging
import tempfile
from pathlib import Path, PurePath
from statistics import mean
from typing import List, Optional, Set

import pandas as pd

from ..common.corpus import load_corpus, tokenize_corpus, write_corpus
from ..common.environment import SIL_NLP_ENV
from .config import get_aligner
from .lexicon import Lexicon
from .machine_aligner import MachineAligner

LOGGER = logging.getLogger(__name__)


def get_experiment_dirs(exp_pattern: str) -> List[Path]:
    exp_dirs: List[Path] = []
    for path in SIL_NLP_ENV.align_experiments_dir.glob(str(PurePath(exp_pattern) / "**" / "config.yml")):
        dir = path.parent
        if len(list(dir.rglob("config.yml"))) == 1:
            exp_dirs.append(dir)
    return exp_dirs


def get_experiment_name(exp_dir: Path) -> str:
    return exp_dir.as_posix()[len(SIL_NLP_ENV.align_experiments_dir.as_posix()) + 1 :]


def compute_alignment_score(
    direct_lexicon: Lexicon,
    inverse_lexicon: Optional[Lexicon],
    src_sentence: str,
    trg_sentence: str,
    alignment: str,
) -> float:
    pairs = alignment.strip().split(" ")
    src_words = src_sentence.strip().split(" ")
    trg_words = trg_sentence.strip().split(" ")
    probs: List[float] = []
    unaligned_trg_indices: Set[int] = set(range(len(trg_words)))
    unaligned_src_indices: Set[int] = set(range(len(src_words)))
    for pair in pairs:
        if pair != "" and pair != "\n":
            prop_parts = pair.split(":")
            parts = prop_parts[0].split("-")
            i = int(parts[0])
            j = int(parts[1])
            unaligned_src_indices.discard(i)
            unaligned_trg_indices.discard(j)
            if len(prop_parts) > 1:
                # the probability is in the alignment itself
                probs.append(float(prop_parts[1]))
            else:
                # grab the probability from the lexicons
                src_word = src_words[i]
                trg_word = trg_words[j]
                direct_prob = max(direct_lexicon[src_word, trg_word], 1e-9)
                inverse_prob = max(inverse_lexicon[trg_word, src_word], 1e-9) if inverse_lexicon else 0
                prob = max(direct_prob, inverse_prob)
                probs.append(prob)
        else:
            LOGGER.warning(
                f"No pairs in alignment! src >>{src_sentence}<< trg >>{trg_sentence}<< alignment >>{alignment}<<"
            )

    for j in unaligned_trg_indices:
        probs.append(max(direct_lexicon["NULL", trg_words[j]], 1e-9))

    for i in unaligned_src_indices:
        if inverse_lexicon:
            probs.append(max(inverse_lexicon["NULL", src_words[i]], 1e-9))
        else:
            probs.append(max(direct_lexicon[src_words[i], "NULL"], 1e-9))

    return mean(probs) if len(probs) > 0 else 0


def add_alignment_scores(corpus: pd.DataFrame, aligner_id: str = "fast_align") -> None:
    with tempfile.TemporaryDirectory() as td:
        src_path = Path(td) / "src-input.txt"
        trg_path = Path(td) / "trg-input.txt"
        write_corpus(src_path, corpus["source"])
        write_corpus(trg_path, corpus["target"])
        scores = compute_alignment_scores(src_path, trg_path, aligner_id)
        corpus["score"] = scores


def compute_alignment_scores(
    src_input_path: Path, trg_input_path: Path, aligner_id: str = "fast_align", sym_align_path: Path = None
) -> List[float]:
    # Check for alignable pairs
    src_sents = list(load_corpus(src_input_path))
    trg_sents = list(load_corpus(trg_input_path))
    if not any(len(s) > 0 and len(t) > 0 for s, t in zip(src_sents, trg_sents)):
        LOGGER.warning("No pairs to align.")
        return [0 for _ in src_sents]

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        src_tok_output_path = temp_dir / "tokenize-src-output.txt"
        trg_tok_output_path = temp_dir / "tokenize-trg-output.txt"

        tokenize_corpus(src_input_path, src_tok_output_path)
        tokenize_corpus(trg_input_path, trg_tok_output_path)

        aligner = get_aligner(aligner_id, temp_dir)
        if sym_align_path is None:
            sym_align_path = temp_dir / "sym-align.txt"
        aligner.train(src_tok_output_path, trg_tok_output_path)
        if isinstance(aligner, MachineAligner):
            aligner.align(sym_align_path, export_probabilities=True)
        else:
            aligner.align(sym_align_path)

        direct_lexicon = aligner.get_direct_lexicon(include_special_tokens=True)
        if aligner.has_inverse_model:
            inverse_lexicon = aligner.get_inverse_lexicon(include_special_tokens=True)
        else:
            inverse_lexicon = None

        scores: List[float] = []
        with (
            src_tok_output_path.open("r", encoding="utf-8") as src_tok_output_file,
            trg_tok_output_path.open("r", encoding="utf-8") as trg_tok_output_file,
            sym_align_path.open("r", encoding="utf-8") as sym_align_file,
        ):
            for src_sentence, trg_sentence, alignment in zip(src_tok_output_file, trg_tok_output_file, sym_align_file):
                if src_sentence.strip() == "" or trg_sentence.strip() == "":
                    scores.append(-1)
                else:
                    scores.append(
                        compute_alignment_score(direct_lexicon, inverse_lexicon, src_sentence, trg_sentence, alignment)
                    )
        return scores
