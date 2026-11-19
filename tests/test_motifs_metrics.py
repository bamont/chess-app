import math

import pytest

np = pytest.importorskip("numpy")

from motifs.metrics import (  # noqa: E402
    ThemeResult,
    average_precision,
    best_f1,
    evaluate_scores,
    f1_at_threshold,
    format_results,
    mean_average_precision,
)

TRUTH = np.array([1, 0, 1, 0])
SCORES = np.array([0.9, 0.8, 0.7, 0.1])


def test_a_perfect_ranking_has_an_average_precision_of_one():
    assert average_precision(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(1.0)


def test_average_precision_rewards_early_hits():
    assert average_precision(TRUTH, SCORES) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_without_positives_is_undefined():
    assert math.isnan(average_precision(np.zeros(4, dtype=int), SCORES))


def test_best_f1_finds_the_best_cut():
    f1, threshold = best_f1(TRUTH, SCORES)
    assert f1 == pytest.approx(0.8)
    assert threshold == pytest.approx(0.7)


def test_best_f1_without_positives():
    assert best_f1(np.zeros(4, dtype=int), SCORES) == (0.0, 0.5)


def test_f1_at_a_fixed_threshold():
    assert f1_at_threshold(TRUTH, SCORES, 0.75) == pytest.approx(0.5)
    assert f1_at_threshold(TRUTH, SCORES, 0.05) == pytest.approx(4.0 / 6.0)
    assert f1_at_threshold(TRUTH, SCORES, 0.95) == 0.0


def test_scores_are_evaluated_for_every_theme():
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
    scores = np.array([[0.9, 0.1], [0.8, 0.9], [0.7, 0.2], [0.1, 0.3]])
    results = evaluate_scores(labels, scores, ("fork", "pin"))
    assert [result.name for result in results] == ["fork", "pin"]
    assert [result.positives for result in results] == [2, 1]
    assert results[0].prevalence == pytest.approx(0.5)
    assert results[1].average_precision == pytest.approx(1.0)
    assert results[0].threshold == pytest.approx(0.7)


def test_fixed_thresholds_replace_the_best_cut():
    labels = np.array([[1], [0], [1], [0]])
    scores = np.array([[0.9], [0.8], [0.7], [0.1]])
    result = evaluate_scores(labels, scores, ("fork",), thresholds=[0.75])[0]
    assert result.threshold == 0.75
    assert result.f1 == pytest.approx(0.5)


def test_mean_average_precision_ignores_undefined_themes():
    results = [
        ThemeResult("a", 3, 0.3, 0.5, 0.4, 0.5),
        ThemeResult("b", 0, 0.0, float("nan"), 0.0, 0.5),
        ThemeResult("c", 2, 0.2, 0.7, 0.6, 0.5),
    ]
    assert mean_average_precision(results) == pytest.approx(0.6)
    assert mean_average_precision([]) == 0.0


def test_the_results_table_lists_every_theme():
    results = [ThemeResult("fork", 3, 0.3, 0.5, 0.4, 0.25)]
    text = format_results(results)
    assert "fork" in text
    assert "mean average precision: 0.500" in text