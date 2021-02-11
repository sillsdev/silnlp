import glob
import math
import os
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from rbo import RankingSimilarity
from nltk.translate import Alignment

from ..common.corpus import load_corpus
from ..common.verse_ref import VerseRef
from .config import get_aligner_name
from .lexicon import Lexicon


def compute_aer(alignments: Iterable[Alignment], references: Iterable[Alignment]) -> float:
    a_count, s_count, pa_count, sa_count = get_alignment_counts(alignments, references)
    if s_count + a_count == 0:
        return 0
    return 1 - ((pa_count + sa_count) / (s_count + a_count))


def compute_f_score(
    alignments: Iterable[Alignment], references: Iterable[Alignment], alpha: float = 0.5
) -> Tuple[float, float, float]:
    a_count, s_count, pa_count, sa_count = get_alignment_counts(alignments, references)
    precision = 1 if a_count == 0 else pa_count / a_count
    recall = 1 if s_count == 0 else sa_count / s_count
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


def load_alignments(input_file_path: str) -> List[Alignment]:
    alignments: List[Alignment] = []
    for line in load_corpus(input_file_path):
        if line.startswith("#"):
            continue
        alignments.append(Alignment.fromstring(line))
    return alignments


def load_all_alignments(root_dir: str) -> Dict[str, List[Alignment]]:
    results: Dict[str, List[Alignment]] = {}
    for alignments_path in glob.glob(os.path.join(root_dir, "alignments.*.txt")):
        file_name = os.path.basename(alignments_path)
        parts = file_name.split(".")
        id = parts[1]

        alignments = load_alignments(alignments_path)
        results[id] = alignments
    return results


def load_all_lexicons(root_dir: str) -> Dict[str, Lexicon]:
    results: Dict[str, Lexicon] = {}
    for lexicon_path in glob.glob(os.path.join(root_dir, "lexicon.*.txt")):
        file_name = os.path.basename(lexicon_path)
        parts = file_name.split(".")
        id = parts[1]

        lexicon = Lexicon.load(lexicon_path)
        results[id] = lexicon
    return results


def load_vrefs(vref_file_path: str) -> List[VerseRef]:
    vrefs: List[VerseRef] = []
    for line in load_corpus(vref_file_path):
        vrefs.append(VerseRef.from_bbbcccvvv(int(line)))
    return vrefs


def filter_alignments_by_book(vrefs: List[VerseRef], alignments: List[Alignment], books: Set[int]) -> List[Alignment]:
    if len(books) == 0:
        return alignments

    results: List[Alignment] = []
    for vref, alignment in zip(vrefs, alignments):
        if vref.book_num in books:
            results.append(alignment)
    return results


def filter_alignments_by_index(alignments: List[Alignment], indices: List[int]) -> List[Alignment]:
    if len(indices) == 0:
        return alignments

    results: List[Alignment] = []
    for index in indices:
        results.append(alignments[index])
    return results


def compute_alignment_metrics(
    vrefs: List[VerseRef],
    all_alignments: Dict[str, List[Alignment]],
    label: str,
    books: Set[int] = set(),
    test_size: Optional[int] = None,
) -> pd.DataFrame:
    references = all_alignments["gold"]
    references = filter_alignments_by_book(vrefs, references, books)
    test_indices: List[int] = []
    if test_size is not None and len(references) > test_size:
        test_indices = random.sample(range(len(references)), test_size)
    references = filter_alignments_by_index(references, test_indices)

    aligner_names: List[str] = []
    aers: List[float] = []
    f_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    if len(references) > 0:
        for aligner_id, alignments in all_alignments.items():
            if aligner_id == "gold":
                continue

            aligner_name = get_aligner_name(aligner_id)
            alignments = filter_alignments_by_book(vrefs, alignments, books)
            alignments = filter_alignments_by_index(alignments, test_indices)

            aer = compute_aer(alignments, references)
            f_score, precision, recall = compute_f_score(alignments, references)

            aligner_names.append(aligner_name)
            aers.append(aer)
            f_scores.append(f_score)
            precisions.append(precision)
            recalls.append(recall)

    return pd.DataFrame(
        {"AER": aers, "F-Score": f_scores, "Precision": precisions, "Recall": recalls},
        columns=["AER", "F-Score", "Precision", "Recall"],
        index=pd.MultiIndex.from_tuples(map(lambda a: (label, a), aligner_names), names=["Book", "Model"]),
    )


def compute_precision_at_k(lexicon: Lexicon, reference: Lexicon, k: int) -> float:
    relevant_count: int = 0
    total_count: int = 0

    for src_word in lexicon.source_words:
        ref_trg_words: Set[str] = set(reference.get_target_words(src_word))

        trg_words = list(lexicon.get_target_words(src_word))
        for trg_word in trg_words[:k]:
            total_count += 1
            if trg_word in ref_trg_words:
                relevant_count += 1

    return 1 if total_count == 0 else relevant_count / total_count


def compute_recall_at_k(lexicon: Lexicon, reference: Lexicon, k: int) -> float:
    relevant_count: int = 0
    total_count: int = 0

    for src_word in reference.source_words:
        ref_trg_words: Set[str] = set(reference.get_target_words(src_word))

        trg_words = list(lexicon.get_target_words(src_word))
        for trg_word in trg_words[:k]:
            if trg_word in ref_trg_words:
                relevant_count += 1
        total_count += min(len(ref_trg_words), k)

    return 1 if total_count == 0 else relevant_count / total_count


def compute_mean_avg_precision(lexicon: Lexicon, reference: Lexicon) -> float:
    ap_sum: float = 0
    src_word_count: int = 0

    for src_word in lexicon.source_words:
        ref_trg_words: Set[str] = set(reference.get_target_words(src_word))

        relevant_count: int = 0
        total_count: int = 0
        pak_sum: float = 0
        trg_words = list(lexicon.get_target_words(src_word))
        for trg_word in trg_words:
            total_count += 1
            if trg_word in ref_trg_words:
                relevant_count += 1
                pak_sum += relevant_count / total_count
        ap_sum += pak_sum / total_count
        src_word_count += 1

    return 1 if src_word_count == 0 else ap_sum / src_word_count


def compute_dcg(scores: List[float]) -> float:
    return sum(map(lambda t: ((2 ** t[1]) - 1) / math.log2(t[0] + 2), enumerate(scores)))


def compute_ndcg(lexicon: Lexicon, reference: Lexicon) -> float:
    dcg_sum: float = 0
    idcg_sum: float = 0
    for src_word in lexicon.source_words:
        scores = list(map(lambda tw: reference[src_word, tw], lexicon.get_target_words(src_word)))
        dcg = compute_dcg(scores)
        dcg_sum += dcg
        idcg = compute_dcg(sorted(scores, reverse=True))
        idcg_sum += idcg
    return 1 if idcg_sum == 0 else dcg_sum / idcg_sum


def compute_rbo(lexicon: Lexicon, reference: Lexicon) -> float:
    rbo_sum: float = 0
    src_word_count: int = 0
    src_words: Set[str] = set(lexicon.source_words)
    src_words.update(reference.source_words)
    for src_word in src_words:
        words = list(lexicon.get_target_words(src_word))
        ref_words = list(reference.get_target_words(src_word))
        rbo_sum += RankingSimilarity(words, ref_words).rbo_ext()
        src_word_count += 1
    return 1 if src_word_count == 0 else rbo_sum / src_word_count


def compute_f_score_at_k(lexicon: Lexicon, reference: Lexicon, k: int) -> Tuple[float, float, float]:
    precision_at_k = compute_precision_at_k(lexicon, reference, k)
    recall_at_k = compute_recall_at_k(lexicon, reference, k)
    f_score_at_k = 2 * ((precision_at_k * recall_at_k) / (precision_at_k + recall_at_k))
    return (f_score_at_k, precision_at_k, recall_at_k)


def compute_lexicon_metrics(root_dir: str, all_lexicons: Dict[str, Lexicon]) -> pd.DataFrame:
    reference = all_lexicons["gold"]

    aligner_names: List[str] = []
    f_score_at_1: List[float] = []
    precision_at_1: List[float] = []
    recall_at_1: List[float] = []
    f_score_at_3: List[float] = []
    precision_at_3: List[float] = []
    recall_at_3: List[float] = []
    mean_avg_precision: List[float] = []
    ndcg: List[float] = []
    rbo: List[float] = []
    for aligner_id, lexicon in all_lexicons.items():
        if aligner_id == "gold":
            continue

        aligner_name = get_aligner_name(aligner_id)
        aligner_names.append(aligner_name)

        f1, p1, r1 = compute_f_score_at_k(lexicon, reference, 1)
        f_score_at_1.append(f1)
        precision_at_1.append(p1)
        recall_at_1.append(r1)

        f3, p3, r3 = compute_f_score_at_k(lexicon, reference, 3)
        f_score_at_3.append(f3)
        precision_at_3.append(p3)
        recall_at_3.append(r3)

        mean_avg_precision.append(compute_mean_avg_precision(lexicon, reference))
        ndcg.append(compute_ndcg(lexicon, reference))
        rbo.append(compute_rbo(lexicon, reference))

    return pd.DataFrame(
        {
            "F-Score@1": f_score_at_1,
            "Precision@1": precision_at_1,
            "Recall@1": recall_at_1,
            "F-Score@3": f_score_at_3,
            "Precision@3": precision_at_3,
            "Recall@3": recall_at_3,
            "MAP": mean_avg_precision,
            "NDCG": ndcg,
            "RBO": rbo,
        },
        columns=["F-Score@1", "Precision@1", "Recall@1", "F-Score@3", "Precision@3", "Recall@3", "MAP", "NDCG", "RBO"],
        index=pd.MultiIndex.from_tuples(map(lambda a: ("ALL", a), aligner_names), names=["Book", "Model"]),
    )

