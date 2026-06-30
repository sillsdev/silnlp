import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

SCHEMA_VERSION = "0.1"
DEFAULT_NUM_BUCKETS = 100


@dataclass
class LinearRegressionResult:
    version: str
    slope: float
    intercept: float

    def toJSON(self) -> str:
        return json.dumps({"version": self.version, "slope": self.slope, "intercept": self.intercept}, indent=2)


class PointWeightingScheme:
    def weight_points(self, _x: List[float], _y: List[float]) -> List[float]: ...


class UniformPointWeightingScheme(PointWeightingScheme):
    def weight_points(self, x: List[float], y: List[float], **kwargs) -> List[float]:
        return [1.0] * len(x)


class InverseDensityPointWeightingScheme(PointWeightingScheme):
    def weight_points(self, _x: List[float], y: List[float], num_buckets: int = DEFAULT_NUM_BUCKETS) -> List[float]:
        density, _ = np.histogram(y, bins=num_buckets, range=(0, 100), density=True)
        bin_edges = np.linspace(0, 100, num=num_buckets + 1)
        weights = []
        for y_instance in y:
            bin_index = np.searchsorted(bin_edges, y_instance, side="right") - 1
            bin_index = min(max(bin_index, 0), len(density) - 1)
            weights.append(1 / (density[bin_index] * len(y) + 1e-6))
        return weights


def perform_simple_linear_regression(x: List[float], y: List[float]) -> LinearRegressionResult:
    slope, intercept = linregress(x, y)[:2]
    slope = round(slope, 4)
    intercept = round(intercept, 4)
    return LinearRegressionResult(version=SCHEMA_VERSION, slope=slope, intercept=intercept)


def linear_model(x, slope, intercept):
    return slope * x + intercept


def perform_weighted_linear_regression(
    x: List[float],
    y: List[float],
    point_weighting_scheme: PointWeightingScheme = UniformPointWeightingScheme(),
) -> LinearRegressionResult:
    weights = point_weighting_scheme.weight_points(x, y)
    sigmas = [1 / np.sqrt(w) if w > 0 else np.inf for w in weights]
    popt, _ = curve_fit(linear_model, x, y, sigma=sigmas, absolute_sigma=True)
    slope, intercept = popt
    slope = round(slope, 4)
    intercept = round(intercept, 4)
    return LinearRegressionResult(version=SCHEMA_VERSION, slope=slope, intercept=intercept)


class BootstrapSampler:
    def __init__(self, x: List[float], y: List[float]) -> None:
        self.x = x
        self.y = y

    def sample(self, num_samples: int) -> Tuple[List[float], List[float]]:
        n = len(self.x)
        sampled_x = []
        sampled_y = []
        for _ in range(num_samples):
            indices = np.random.choice(n, size=n, replace=True)
            sampled_x.extend([self.x[i] for i in indices])
            sampled_y.extend([self.y[i] for i in indices])
        return sampled_x, sampled_y


class LinearRegressionResultFilter:
    def filter(self, _results: List[LinearRegressionResult]) -> List[LinearRegressionResult]: ...


class SlopePercentileLinearRegressionResultFilter(LinearRegressionResultFilter):
    def __init__(self, percentile: float = 90.0) -> None:
        self.percentile = percentile

    def filter(self, results: List[LinearRegressionResult]) -> List[LinearRegressionResult]:
        slopes = [result.slope for result in results]
        threshold = np.percentile(slopes, self.percentile)
        return [result for result in results if result.slope >= threshold]


class LinearRegressionResultAggregator:
    def aggregate(self, _results: List[LinearRegressionResult]) -> LinearRegressionResult: ...


class AverageLinearRegressionResultAggregator(LinearRegressionResultAggregator):
    def aggregate(self, results: List[LinearRegressionResult]) -> LinearRegressionResult:
        assert len(results) > 0, "No results to aggregate"
        avg_slope = round(np.mean([result.slope for result in results]), 4)
        avg_intercept = round(np.mean([result.intercept for result in results]), 4)
        return LinearRegressionResult(version=SCHEMA_VERSION, slope=avg_slope, intercept=avg_intercept)


def perform_enhanced_linear_regression(x: List[float], y: List[float]) -> LinearRegressionResult:
    sampler = BootstrapSampler(x, y)
    weighting_scheme = InverseDensityPointWeightingScheme()

    bootstrap_results = []
    for _ in range(100):
        sampled_x, sampled_y = sampler.sample(num_samples=len(x))
        result = perform_weighted_linear_regression(sampled_x, sampled_y, point_weighting_scheme=weighting_scheme)
        bootstrap_results.append(result)

    filtered_results = SlopePercentileLinearRegressionResultFilter(percentile=90.0).filter(bootstrap_results)
    return AverageLinearRegressionResultAggregator().aggregate(filtered_results)
