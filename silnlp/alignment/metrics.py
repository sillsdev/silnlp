import glob
import os
import random
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from nltk.translate import Alignment

from nlp.alignment.config import get_aligner
from nlp.common.corpus import load_corpus


def compute_aer(alignments: Iterable[Alignment], references: Iterable[Alignment]) -> float:
    a_count, s_count, pa_count, sa_count = get_alignment_counts(alignments, references)
    return 1 - ((pa_count + sa_count) / (s_count + a_count))


def compute_f_score(
    alignments: Iterable[Alignment], references: Iterable[Alignment], alpha: float = 0.5
) -> Tuple[float, float, float]:
    a_count, s_count, pa_count, sa_count = get_alignment_counts(alignments, references)
    precision = pa_count / a_count
    recall = sa_count / s_count
    f_score = 1 / ((alpha / precision) + ((1 - alpha) / recall))
    return (f_score, precision, recall)


def get_alignment_counts(alignments: Iterable[Alignment], references: Iterable[Alignment]) -> Tuple[int, int, int, int]:
    a_count = 0
    s_count = 0
    pa_count = 0
    sa_count = 0
    for alignment, reference in zip(alignments, references):
        a_count += len(alignment)
        for wp in reference:
            if len(wp) < 3 or wp[2]:
                s_count += 1
                if (wp[0], wp[1]) in alignment:
                    sa_count += 1
                    pa_count += 1
            elif (wp[0], wp[1]) in alignment:
                pa_count += 1
    return (a_count, s_count, pa_count, sa_count)


def load_alignments(input_file: str) -> List[Alignment]:
    alignments: List[Alignment] = []
    for line in load_corpus(input_file):
        if line.startswith("#"):
            continue
        alignments.append(Alignment.fromstring(line))
    return alignments


def filter_alignments(alignments: List[Alignment], indices: List[int]) -> List[Alignment]:
    results: List[Alignment] = []
    for index in indices:
        results.append(alignments[index])
    return results


def compute_metrics(root_dir: str, test_size: Optional[int] = None) -> pd.DataFrame:
    ref_file_path = os.path.join(root_dir, "alignments.gold.txt")
    references = load_alignments(ref_file_path)

    test_indices: Optional[List[int]] = None
    if test_size is not None:
        test_indices = random.sample(range(len(references)), test_size)
        references = filter_alignments(references, test_indices)

    aligner_names: List[str] = []
    aers: List[float] = []
    f_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    for alignments_path in glob.glob(os.path.join(root_dir, "alignments.*.txt")):
        if alignments_path == ref_file_path:
            continue
        file_name = os.path.basename(alignments_path)
        parts = file_name.split(".")
        id = parts[1]
        aligner = get_aligner(id, root_dir)

        alignments = load_alignments(alignments_path)
        if test_indices is not None:
            alignments = filter_alignments(alignments, test_indices)

        aer = compute_aer(alignments, references)
        f_score, precision, recall = compute_f_score(alignments, references)

        aligner_names.append(aligner.name)
        aers.append(aer)
        f_scores.append(f_score)
        precisions.append(precision)
        recalls.append(recall)

    return pd.DataFrame(
        {"AER": aers, "F-Score": f_scores, "Precision": precisions, "Recall": recalls},
        columns=["AER", "F-Score", "Precision", "Recall"],
        index=aligner_names,
    )
