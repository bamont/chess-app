import pytest

from ui.eval_curve import curve_index_at, curve_points, curve_values, curve_x


def test_the_scores_become_white_shares():
    values = curve_values([0, 9980, -9980])
    assert values[0] == pytest.approx(0.5)
    assert values[1] == 1.0
    assert values[2] == 0.0


def test_a_winning_position_for_white_is_drawn_high_unless_the_board_is_flipped():
    normal = curve_points([1.0, 0.5, 0.0], 10, 100, 200, 60, False)
    assert normal[0] == (10, 100)
    assert normal[1] == (110, 130)
    assert normal[2] == (210, 160)
    flipped = curve_points([1.0, 0.5, 0.0], 10, 100, 200, 60, True)
    assert flipped[0] == (10, 160)
    assert flipped[2] == (210, 100)


def test_one_value_is_drawn_at_the_left_edge():
    assert curve_points([0.5], 10, 100, 200, 60, False) == [(10, 130)]


def test_no_value_gives_no_point():
    assert curve_points([], 10, 100, 200, 60, False) == []


def test_the_cursor_follows_the_index():
    assert curve_x(0, 10, 200, 5) == 10
    assert curve_x(4, 10, 200, 5) == 210
    assert curve_x(2, 10, 200, 5) == 110
    assert curve_x(0, 10, 200, 1) == 10


def test_a_click_gives_the_nearest_position():
    assert curve_index_at(10, 10, 200, 5) == 0
    assert curve_index_at(210, 10, 200, 5) == 4
    assert curve_index_at(110, 10, 200, 5) == 2
    assert curve_index_at(70, 10, 200, 5) == 1
    assert curve_index_at(10, 10, 200, 1) == 0


def test_a_click_outside_the_curve_gives_nothing():
    assert curve_index_at(9, 10, 200, 5) is None
    assert curve_index_at(211, 10, 200, 5) is None
    assert curve_index_at(50, 10, 200, 0) is None


def test_every_position_can_be_reached_by_clicking_on_its_own_point():
    count = 138
    for index in range(count):
        assert curve_index_at(curve_x(index, 10, 296, count), 10, 296, count) == index