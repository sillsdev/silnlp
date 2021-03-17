import os
import tempfile
from statistics import mean
from typing import List, Set

import pandas as pd

from ..common.corpus import tokenize_corpus, write_corpus
from ..common.environment import ALIGN_EXPERIMENTS_DIR
from .lexicon import Lexicon
from .machine_aligner import FastAlign


def get_align_exp_dir(exp_name: str) -> str:
    return os.path.join(ALIGN_EXPERIMENTS_DIR, exp_name)


def compute_alignment_score(
    direct_lexicon: Lexicon, inverse_lexicon: Lexicon, src_sentence: str, trg_sentence: str, alignment: str,
) -> float:
    pairs = alignment.split(" ")
    src_words = src_sentence.split(" ")
    trg_words = trg_sentence.split(" ")
    probs: List[float] = []
    unaligned_trg_indices: Set[int] = set(range(len(trg_words)))
    unaligned_src_indices: Set[int] = set(range(len(src_words)))
    for pair in pairs:
        if pair != "":
            parts = pair.split("-")
            i = int(parts[0])
            j = int(parts[1])
            unaligned_src_indices.discard(i)
            unaligned_trg_indices.discard(j)
            src_word = src_words[i]
            trg_word = trg_words[j]
            direct_prob = max(direct_lexicon[src_word, trg_word], 1e-9)
            inverse_prob = max(inverse_lexicon[trg_word, src_word], 1e-9)
            prob = max(direct_prob, inverse_prob)
            probs.append(prob)

    for j in unaligned_trg_indices:
        probs.append(max(direct_lexicon["NULL", trg_words[j]], 1e-9))

    for i in unaligned_src_indices:
        probs.append(max(inverse_lexicon["NULL", src_words[i]], 1e-9))

    return mean(probs)


def add_alignment_scores(corpus: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, "src-input.txt")
        trg_path = os.path.join(td, "trg-input.txt")
        write_corpus(src_path, corpus["source"])
        write_corpus(trg_path, corpus["target"])
        scores = compute_alignment_scores(src_path, trg_path)
        corpus["score"] = scores


def compute_alignment_scores(src_input_path: str, trg_input_path: str) -> List[float]:
    with tempfile.TemporaryDirectory() as td:
        src_tok_output_path = os.path.join(td, "tokenize-src-output.txt")
        trg_tok_output_path = os.path.join(td, "tokenize-trg-output.txt")

        tokenize_corpus(src_input_path, src_tok_output_path)
        tokenize_corpus(trg_input_path, trg_tok_output_path)

        fast_align = FastAlign(td)

        sym_align_path = os.path.join(td, "sym-align.txt")
        fast_align.train(src_tok_output_path, trg_tok_output_path)
        fast_align.align(sym_align_path)

        direct_lexicon = fast_align.get_direct_lexicon(include_special_tokens=True)
        inverse_lexicon = fast_align.get_inverse_lexicon(include_special_tokens=True)

        scores: List[float] = []
        with open(src_tok_output_path, "r", encoding="utf-8") as src_tok_output_file, open(
            trg_tok_output_path, "r", encoding="utf-8"
        ) as trg_tok_output_file, open(sym_align_path, "r", encoding="utf-8") as sym_align_file:
            for src_sentence, trg_sentence, alignment in zip(src_tok_output_file, trg_tok_output_file, sym_align_file):
                scores.append(
                    compute_alignment_score(direct_lexicon, inverse_lexicon, src_sentence, trg_sentence, alignment)
                )
        return scores
