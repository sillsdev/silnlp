import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas as pd
from nltk.translate import Alignment

from ..common.corpus import load_corpus
from ..common.verse_ref import VerseRef
from .config import get_aligner_name
from .lexicon import Lexicon
from .rbo import average_overlap, rbo_ext


def corpus_aer(alignments: Iterable[Alignment], references: Iterable[Alignment]) -> float:
    a_count, s_count, pa_count, sa_count = get_alignment_counts(alignments, references)
    if s_count + a_count == 0:
        return 0
    return 1 - ((pa_count + sa_count) / (s_count + a_count))


def corpus_f_score(
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


def load_alignments(input_file_path: Path) -> List[Alignment]:
    alignments: List[Alignment] = []
    for line in load_corpus(input_file_path):
        if line.startswith("#"):
            continue
        alignments.append(Alignment.fromstring(line))
    return alignments


def load_all_alignments(exp_dir: Path) -> Dict[str, List[Alignment]]:
    results: Dict[str, List[Alignment]] = {}
    for alignments_path in exp_dir.glob("alignments.*.txt"):
        file_name = alignments_path.name
        parts = file_name.split(".")
        id = parts[1]

        alignments = load_alignments(alignments_path)
        results[id] = alignments
    return results


def load_all_lexicons(exp_dir: Path) -> Dict[str, Lexicon]:
    results: Dict[str, Lexicon] = {}
    for lexicon_path in exp_dir.glob("lexicon.*.txt"):
        file_name = lexicon_path.name
        parts = file_name.split(".")
        id = parts[1]

        lexicon = Lexicon.load(lexicon_path)
        results[id] = lexicon
    return results


def load_vrefs(vref_file_path: Path) -> List[VerseRef]:
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
    aer_list: List[float] = []
    f_score_list: List[float] = []
    precision_list: List[float] = []
    recall_list: List[float] = []
    if len(references) > 0:
        for aligner_id, alignments in all_alignments.items():
            if aligner_id == "gold":
                continue

            aligner_name = get_aligner_name(aligner_id)
            alignments = filter_alignments_by_book(vrefs, alignments, books)
            alignments = filter_alignments_by_index(alignments, test_indices)

            aer = corpus_aer(alignments, references)
            f_score, precision, recall = corpus_f_score(alignments, references)

            aligner_names.append(aligner_name)
            aer_list.append(aer)
            f_score_list.append(f_score)
            precision_list.append(precision)
            recall_list.append(recall)

    return pd.DataFrame(
        {"AER": aer_list, "F-Score": f_score_list, "Precision": precision_list, "Recall": recall_list},
        columns=["AER", "F-Score", "Precision", "Recall"],
        index=pd.MultiIndex.from_tuples(map(lambda a: (label, a), aligner_names), names=["Book", "Model"]),
    )


def corpus_precision_at_k(lexicon: Lexicon, reference: Lexicon, k: int) -> float:
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


def corpus_recall_at_k(lexicon: Lexicon, reference: Lexicon, k: int) -> float:
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


def corpus_mean_avg_precision(lexicon: Lexicon, reference: Lexicon) -> float:
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


def group_by_rank(word_probs: Iterable[Tuple[str, float]]) -> List[Union[str, Set[str]]]:
    last_prob: float = -1
    ranks: List[Union[str, Set[str]]] = []
    for word, prob in word_probs:
        if prob == last_prob:
            last_rank = ranks[-1]
            if isinstance(last_rank, str):
                last_rank = {last_rank, word}
                ranks[-1] = last_rank
            else:
                last_rank.add(word)
        else:
            ranks.append(word)
        last_prob = prob
    return ranks


def corpus_average_overlap_at_k(lexicon: Lexicon, reference: Lexicon, k: int) -> float:
    rbo_sum: float = 0
    src_word_count: int = 0
    src_words: Set[str] = set(lexicon.source_words)
    src_words.update(reference.source_words)
    for src_word in src_words:
        words = group_by_rank(lexicon.get_target_word_probs(src_word))
        ref_words = group_by_rank(reference.get_target_word_probs(src_word))
        rbo_sum += average_overlap(words, ref_words, depth=k)
        src_word_count += 1
    return 1 if src_word_count == 0 else rbo_sum / src_word_count


def corpus_rbo_ext(lexicon: Lexicon, reference: Lexicon, p: float = 0.9) -> float:
    rbo_sum: float = 0
    src_word_count: int = 0
    src_words: Set[str] = set(lexicon.source_words)
    src_words.update(reference.source_words)
    for src_word in src_words:
        words = group_by_rank(lexicon.get_target_word_probs(src_word))
        ref_words = group_by_rank(reference.get_target_word_probs(src_word))
        rbo_sum += rbo_ext(words, ref_words, p)
        src_word_count += 1
    return 1 if src_word_count == 0 else rbo_sum / src_word_count


def corpus_f_score_at_k(lexicon: Lexicon, reference: Lexicon, k: int) -> Tuple[float, float, float]:
    precision_at_k = corpus_precision_at_k(lexicon, reference, k)
    recall_at_k = corpus_recall_at_k(lexicon, reference, k)
    f_score_at_k = 2 * ((precision_at_k * recall_at_k) / (precision_at_k + recall_at_k))
    return (f_score_at_k, precision_at_k, recall_at_k)


def compute_lexicon_metrics(all_lexicons: Dict[str, Lexicon]) -> pd.DataFrame:
    reference = all_lexicons["gold"]

    aligner_names: List[str] = []
    f_score_at_1_list: List[float] = []
    precision_at_1_list: List[float] = []
    recall_at_1_list: List[float] = []
    f_score_at_3_list: List[float] = []
    precision_at_3_list: List[float] = []
    recall_at_3_list: List[float] = []
    mean_avg_precision_list: List[float] = []
    rbo_list: List[float] = []
    ao_at_1_list: List[float] = []
    for aligner_id, lexicon in all_lexicons.items():
        if aligner_id == "gold":
            continue

        aligner_name = get_aligner_name(aligner_id)
        aligner_names.append(aligner_name)

        ao_at_1_list.append(corpus_average_overlap_at_k(lexicon, reference, 1))
        rbo_list.append(corpus_rbo_ext(lexicon, reference))

        f_score_at_1, precision_at_1, recall_at_1 = corpus_f_score_at_k(lexicon, reference, 1)
        f_score_at_1_list.append(f_score_at_1)
        precision_at_1_list.append(precision_at_1)
        recall_at_1_list.append(recall_at_1)

        f_score_at_3, precision_at_3, recall_at_3 = corpus_f_score_at_k(lexicon, reference, 3)
        f_score_at_3_list.append(f_score_at_3)
        precision_at_3_list.append(precision_at_3)
        recall_at_3_list.append(recall_at_3)

        mean_avg_precision_list.append(corpus_mean_avg_precision(lexicon, reference))

    return pd.DataFrame(
        {
            "F-Score@1": f_score_at_1_list,
            "Precision@1": precision_at_1_list,
            "Recall@1": recall_at_1_list,
            "F-Score@3": f_score_at_3_list,
            "Precision@3": precision_at_3_list,
            "Recall@3": recall_at_3_list,
            "MAP": mean_avg_precision_list,
            "AO@1": ao_at_1_list,
            "RBO": rbo_list,
        },
        columns=["F-Score@1", "Precision@1", "Recall@1", "F-Score@3", "Precision@3", "Recall@3", "MAP", "AO@1", "RBO"],
        index=pd.MultiIndex.from_tuples(map(lambda a: ("ALL", a), aligner_names), names=["Book", "Model"]),
    )
