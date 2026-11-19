import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThemeResult:
    name: str
    positives: int
    prevalence: float
    average_precision: float
    f1: float
    threshold: float


def average_precision(truth: np.ndarray, scores: np.ndarray) -> float:
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = truth[order].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / positives)


def best_f1(truth: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(truth.sum())
    if positives == 0:
        return 0.0, 0.5
    order = np.argsort(-scores, kind="stable")
    hits = truth[order].astype(np.float64)
    true_positives = np.cumsum(hits)
    precision = true_positives / np.arange(1, len(hits) + 1)
    recall = true_positives / positives
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return float(f1[best]), float(scores[order][best])


def f1_at_threshold(truth: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    true_positives = int(np.logical_and(predicted, truth.astype(bool)).sum())
    denominator = int(predicted.sum()) + int(truth.sum())
    return 2.0 * true_positives / denominator if denominator else 0.0


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    themes: Sequence[str],
    thresholds: Sequence[float] | None = None,
) -> list[ThemeResult]:
    results = []
    total = len(labels)
    for index, name in enumerate(themes):
        truth = labels[:, index]
        positives = int(truth.sum())
        if thresholds is None:
            f1, threshold = best_f1(truth, scores[:, index])
        else:
            threshold = float(thresholds[index])
            f1 = f1_at_threshold(truth, scores[:, index], threshold)
        results.append(
            ThemeResult(
                name,
                positives,
                positives / total if total else 0.0,
                average_precision(truth, scores[:, index]),
                f1,
                threshold,
            )
        )
    return results


def mean_average_precision(results: Sequence[ThemeResult]) -> float:
    values = [result.average_precision for result in results if not math.isnan(result.average_precision)]
    return sum(values) / len(values) if values else 0.0


def format_results(results: Sequence[ThemeResult]) -> str:
    lines = [f"{'theme':<20}{'positives':>10}{'prevalence':>12}{'AP':>8}{'F1':>8}{'threshold':>11}"]
    for result in results:
        lines.append(
            f"{result.name:<20}{result.positives:>10}{result.prevalence:>12.3f}"
            f"{result.average_precision:>8.3f}{result.f1:>8.3f}{result.threshold:>11.3f}"
        )
    lines.append(f"mean average precision: {mean_average_precision(results):.3f}")
    return "\n".join(lines)