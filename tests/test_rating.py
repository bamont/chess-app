import pytest

from tools.rating import (
    LevelResult,
    elo_bound,
    estimate_rating,
    expected_score,
    rating_standard_error,
    wilson_interval,
)


def test_expected_score():
    assert expected_score(0) == pytest.approx(0.5)
    assert expected_score(400) == pytest.approx(10 / 11)
    assert expected_score(-400) == pytest.approx(1 / 11)


def test_level_result_counts_points():
    result = LevelResult(level=1500, wins=3, draws=2, losses=5)
    assert result.games == 10
    assert result.points == 4.0
    assert result.score == pytest.approx(0.4)


def test_wilson_interval_surrounds_the_score():
    low, high = wilson_interval(0.5, 20)
    assert 0 <= low < 0.5 < high <= 1


def test_wilson_interval_narrows_with_more_games():
    small = wilson_interval(0.5, 10)
    large = wilson_interval(0.5, 40)
    assert large[1] - large[0] < small[1] - small[0]


def test_wilson_interval_at_the_extremes():
    low, high = wilson_interval(0.0, 20)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert 0 < high < 0.3


def test_estimate_equals_the_level_at_fifty_percent():
    assert estimate_rating([LevelResult(1500, 5, 0, 5)]) == pytest.approx(1500, abs=0.5)


def test_estimate_when_winning():
    assert estimate_rating([LevelResult(1500, 15, 0, 5)]) == pytest.approx(1690.85, abs=1)


def test_estimate_when_losing():
    assert estimate_rating([LevelResult(1500, 5, 0, 15)]) == pytest.approx(1309.15, abs=1)


def test_pooled_estimate_across_two_levels():
    results = [LevelResult(1400, 8, 0, 2), LevelResult(1600, 2, 0, 8)]
    assert estimate_rating(results) == pytest.approx(1500, abs=1)


def test_draws_count_as_half_a_point():
    assert estimate_rating([LevelResult(1500, 0, 10, 0)]) == pytest.approx(1500, abs=0.5)


def test_no_estimate_when_every_game_was_lost_or_won():
    assert estimate_rating([LevelResult(1500, 0, 0, 10)]) is None
    assert estimate_rating([LevelResult(1500, 10, 0, 0)]) is None
    assert estimate_rating([]) is None


def test_standard_error_shrinks_with_more_games():
    small = rating_standard_error([LevelResult(1500, 5, 0, 5)], 1500)
    large = rating_standard_error([LevelResult(1500, 50, 0, 50)], 1500)
    assert large < small


def test_elo_bound_formatting():
    assert elo_bound(0.0) == "-inf"
    assert elo_bound(1.0) == "+inf"
    assert elo_bound(0.5) == "+0"
    assert elo_bound(0.75) == "+191"