import argparse
import glob
import math
import os
from typing import List, Optional, Tuple

import numpy as np

from nlp.alignment.config import ALIGNERS, load_config
from nlp.alignment.metrics import compute_metrics
from nlp.common.environment import align_experiments_dir
from nlp.common.utils import get_align_root_dir, set_seed


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


def get_metrics(exp_name: str, test_size: Optional[int] = None) -> List[float]:
    config = load_config(exp_name)
    set_seed(config["seed"])
    df = compute_metrics(get_align_root_dir(exp_name), test_size)
    metrics = df["F-Score"].to_numpy().tolist()
    assert len(metrics) == len(ALIGNERS)
    return metrics


def compute_similarity(experiments: List[str], base_metrics: List[float], test_size: int) -> Tuple[float, float]:
    metrics: List[float] = []
    for exp_name in experiments:
        metrics.extend(get_metrics(exp_name, test_size))
    similarity = cosine_similarity(base_metrics, metrics)
    correlation = np.corrcoef(base_metrics, metrics)[0, 1]
    return similarity, correlation


def main() -> None:
    parser = argparse.ArgumentParser(description="Finds the optimal size for a gold standard")
    parser.add_argument("testament", help="Testament")
    parser.add_argument("--threshold", type=float, help="Similarity threshold")
    parser.add_argument("--test-size", type=int, help="Test size")
    args = parser.parse_args()

    testament: str = args.testament
    testament = testament.lower()
    experiments: List[str] = []
    base_metrics: List[float] = []
    for path in glob.glob(os.path.join(align_experiments_dir, f"*.{testament}")):
        if os.path.isdir(path):
            exp_name = os.path.basename(path)
            experiments.append(exp_name)
            base_metrics.extend(get_metrics(exp_name))

    test_size: int
    similarity: float
    correlation: float
    if args.test_size is not None:
        test_size = args.test_size
        similarity, correlation = compute_similarity(experiments, base_metrics, test_size)
    else:
        threshold = 0.999
        if args.threshold is not None:
            threshold = args.threshold / 100.0
        similarity = 0.0
        correlation = 0.0
        test_size = 0
        while similarity < threshold:
            test_size += 10
            similarity, correlation = compute_similarity(experiments, base_metrics, test_size)

    print(f"Test size: {test_size}")
    print(f"Similarity: {similarity:.2%}")
    print(f"Correlation: {correlation:.4f}")


if __name__ == "__main__":
    main()
