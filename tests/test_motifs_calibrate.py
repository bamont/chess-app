import pytest

np = pytest.importorskip("numpy")

from motifs.calibrate import (  # noqa: E402
    calibrate,
    false_alarm_rate,
    format_calibration,
    weighted_best_f1,
)
from motifs.metrics import best_f1  # noqa: E402

TRUTH = np.array([1, 0, 1, 0])
SCORES = np.array([0.9, 0.8, 0.7, 0.1])


def test_equal_weights_give_the_usual_best_f1():
    f1, threshold = weighted_best_f1(TRUTH, SCORES, np.ones(4))
    expected_f1, expected_threshold = best_f1(TRUTH, SCORES)
    assert f1 == pytest.approx(expected_f1)
    assert threshold == pytest.approx(expected_threshold)


def test_heavy_negatives_push_the_cut_up():
    truth = np.array([1, 0, 1])
    scores = np.array([0.9, 0.6, 0.55])
    _, light = weighted_best_f1(truth, scores, np.array([1.0, 1.0, 1.0]))
    _, heavy = weighted_best_f1(truth, scores, np.array([1.0, 5.0, 1.0]))
    assert light == pytest.approx(0.55)
    assert heavy == pytest.approx(0.9)


def test_a_theme_without_positives_keeps_a_default_cut():
    assert weighted_best_f1(np.zeros(3), np.array([0.1, 0.2, 0.3]), np.ones(3)) == (0.0, 0.5)


def test_false_alarms_are_counted_on_ordinary_positions_only():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    negatives = np.array([0, 1, 1, 1])
    assert false_alarm_rate(scores, negatives, 0.75) == pytest.approx(1 / 3)
    assert false_alarm_rate(scores, np.zeros(4), 0.5) == 0.0


def test_calibration_raises_a_threshold_that_fires_on_ordinary_positions():
    scores = np.array([[0.9], [0.55], [0.6], [0.2]])
    labels = np.array([[1], [1], [0], [0]])
    negatives = np.array([0, 0, 1, 0])
    result = calibrate(scores, labels, negatives, [0.5], ("fork",), negative_weight=5.0)[0]
    assert result.old_threshold == 0.5
    assert result.new_threshold == pytest.approx(0.9)
    assert result.old_false_alarms == 1.0
    assert result.new_false_alarms == 0.0
    assert result.recall == pytest.approx(0.5)


def test_a_theme_without_positives_keeps_its_threshold():
    scores = np.array([[0.9, 0.2], [0.1, 0.3]])
    labels = np.array([[1, 0], [0, 0]])
    negatives = np.array([0, 1])
    results = calibrate(scores, labels, negatives, [0.5, 0.4], ("fork", "pin"), negative_weight=5.0)
    assert results[1].new_threshold == 0.4


def test_the_calibration_table_lists_every_theme():
    scores = np.array([[0.9], [0.1]])
    results = calibrate(scores, np.array([[1], [0]]), np.array([0, 1]), [0.5], ("fork",), negative_weight=5.0)
    text = format_calibration(results)
    assert "fork" in text
    assert "false alarms" in text