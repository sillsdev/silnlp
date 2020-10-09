from typing import Iterable, Tuple
from nltk.translate import Alignment


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
