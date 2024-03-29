import argparse
import math
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from machine.scripture import get_books

from ..common.utils import set_seed
from .config import ALIGNERS, load_config
from .metrics import compute_alignment_metrics, load_all_alignments, load_vrefs
from .utils import get_experiment_dirs


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot: float = 0
    obs_total: float = 0
    exp_total: float = 0
    for v1, v2 in zip(vec1, vec2):
        dot += v1 * v2
        obs_total += v1 * v1
        exp_total += v2 * v2
    if obs_total == 0 or exp_total == 0:
        return 0
    return dot / (math.sqrt(obs_total) * math.sqrt(exp_total))


def get_metrics(exp_dir: Path, books: Set[int] = set(), test_size: Optional[int] = None) -> List[float]:
    config = load_config(exp_dir)
    set_seed(config["seed"])

    vref_file_path = exp_dir / "refs.txt"
    vrefs = load_vrefs(vref_file_path)

    all_alignments = load_all_alignments(exp_dir)

    df = compute_alignment_metrics(vrefs, all_alignments, "ALL", books, test_size)
    metrics = df["F-Score"].to_numpy().tolist()
    assert len(metrics) == len(ALIGNERS)
    return metrics


def compute_similarity(
    experiments: List[Path], base_metrics: List[float], books: Set[int], test_size: int
) -> Tuple[float, float]:
    metrics: List[float] = []
    for exp_dir in experiments:
        metrics.extend(get_metrics(exp_dir, books, test_size))
    similarity = cosine_similarity(base_metrics, metrics)
    correlation = np.corrcoef(base_metrics, metrics)[0, 1]
    return similarity, correlation


def main() -> None:
    parser = argparse.ArgumentParser(description="Finds the optimal size for a gold standard")
    parser.add_argument("experiments", help="Experiment pattern")
    parser.add_argument("--threshold", type=float, help="Similarity threshold")
    parser.add_argument("--test-size", type=int, help="Test size")
    parser.add_argument("--books", nargs="*", metavar="book", default=[], help="Books")
    args = parser.parse_args()

    testament: str = args.testament
    testament = testament.lower()

    books = get_books(args.books)

    experiments: List[Path] = []
    base_metrics: List[float] = []
    for exp_dir in get_experiment_dirs(args.experiments):
        experiments.append(exp_dir)
        base_metrics.extend(get_metrics(exp_dir))

    test_size: int
    similarity: float
    correlation: float
    if args.test_size is not None:
        test_size = args.test_size
        similarity, correlation = compute_similarity(experiments, base_metrics, books, test_size)
    else:
        threshold = 0.999
        if args.threshold is not None:
            threshold = args.threshold / 100.0
        similarity = 0.0
        correlation = 0.0
        test_size = 0
        while similarity < threshold:
            test_size += 10
            similarity, correlation = compute_similarity(experiments, base_metrics, books, test_size)

    print(f"Test size: {test_size}")
    print(f"Similarity: {similarity:.2%}")
    print(f"Correlation: {correlation:.4f}")


if __name__ == "__main__":
    main()
