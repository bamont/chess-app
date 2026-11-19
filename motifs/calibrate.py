import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from motifs.dataset import VALIDATION, PuzzleSet, load_dataset
from motifs.encoding import unpack_positions
from motifs.infer import MotifModel

SCORING_BATCH = 512


@dataclass(frozen=True)
class ThemeCalibration:
    name: str
    old_threshold: float
    new_threshold: float
    recall: float
    old_false_alarms: float
    new_false_alarms: float


def weighted_best_f1(truth: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float((truth * weights).sum())
    if total == 0.0:
        return 0.0, 0.5
    order = np.argsort(-scores, kind="stable")
    ranked_truth = truth[order].astype(np.float64)
    ranked_weights = weights[order].astype(np.float64)
    true_positives = np.cumsum(ranked_truth * ranked_weights)
    false_positives = np.cumsum((1.0 - ranked_truth) * ranked_weights)
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-12)
    recall = true_positives / total
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return float(f1[best]), float(scores[order][best])


def false_alarm_rate(scores: np.ndarray, negatives: np.ndarray, threshold: float) -> float:
    if not negatives.any():
        return 0.0
    return float((scores[negatives.astype(bool)] >= threshold).mean())


def calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    negatives: np.ndarray,
    old_thresholds: Sequence[float],
    themes: Sequence[str],
    negative_weight: float,
) -> list[ThemeCalibration]:
    weights = np.where(negatives == 1, negative_weight, 1.0)
    results = []
    for index, name in enumerate(themes):
        truth = labels[:, index]
        column = scores[:, index]
        old = float(old_thresholds[index])
        if truth.sum() == 0:
            new = old
        else:
            _, new = weighted_best_f1(truth, column, weights)
        positives = truth.astype(bool)
        recall = float((column[positives] >= new).mean()) if positives.any() else 0.0
        results.append(
            ThemeCalibration(
                name,
                old,
                new,
                recall,
                false_alarm_rate(column, negatives, old),
                false_alarm_rate(column, negatives, new),
            )
        )
    return results


def format_calibration(results: Sequence[ThemeCalibration]) -> str:
    lines = [f"{'theme':<20}{'old thr':>9}{'new thr':>9}{'recall':>8}{'false alarms old':>18}{'new':>8}"]
    for item in results:
        lines.append(
            f"{item.name:<20}{item.old_threshold:>9.3f}{item.new_threshold:>9.3f}{item.recall:>8.3f}"
            f"{100 * item.old_false_alarms:>17.1f}%{100 * item.new_false_alarms:>7.1f}%"
        )
    return "\n".join(lines)


def validation_scores(model: MotifModel, dataset: PuzzleSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(dataset.splits == VALIDATION)
    outputs = []
    for start in range(0, len(indices), SCORING_BATCH):
        batch = indices[start : start + SCORING_BATCH]
        outputs.append(model.scores(unpack_positions(dataset.positions[batch])))
    scores = np.concatenate(outputs) if outputs else np.zeros((0, len(model.themes)))
    return scores, dataset.labels[indices], dataset.negatives[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose motif thresholds that keep false alarms on ordinary positions low.")
    parser.add_argument("model", help=".npz model written by motifs.export")
    parser.add_argument("dataset", help=".npz dataset built with motifs.build --with-negatives")
    parser.add_argument("output", help=".npz model to write, with the new thresholds")
    parser.add_argument(
        "--negative-weight",
        type=float,
        default=5.0,
        help="how many puzzle positions one ordinary position is worth (higher means fewer false alarms)",
    )
    args = parser.parse_args()
    model = MotifModel.load(args.model)
    dataset = load_dataset(args.dataset)
    if dataset.themes != model.themes:
        parser.error("The dataset and the model do not use the same themes")
    scores, labels, negatives = validation_scores(model, dataset)
    if not negatives.any():
        parser.error("The dataset has no negative positions: build it with --with-negatives")
    results = calibrate(scores, labels, negatives, model.thresholds, model.themes, args.negative_weight)
    print(f"{len(scores)} validation positions, {int(negatives.sum())} of them ordinary positions")
    print(format_calibration(results))
    arrays = dict(model.arrays)
    arrays["thresholds"] = np.array([item.new_threshold for item in results], dtype=np.float32)
    np.savez_compressed(args.output, **arrays)
    print(f"model written to {args.output}")


if __name__ == "__main__":
    main()