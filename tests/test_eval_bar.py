import pytest

from ui.eval_bar import bar_label, label_at_top, white_region, white_share


def test_an_equal_position_gives_half_of_the_bar_to_each_side():
    assert white_share(0) == pytest.approx(0.5)
    assert white_region(0, 640, False) == (320, 320)
    assert white_region(0, 640, True) == (0, 320)


def test_the_white_share_grows_with_the_score():
    assert white_share(-300) < white_share(0) < white_share(100) < white_share(900) < 1.0


def test_a_forced_mate_fills_the_bar_for_the_winner():
    assert white_share(9980) == 1.0
    assert white_share(-9980) == 0.0
    assert white_region(9980, 640, False) == (0, 640)
    assert white_region(-9980, 640, False) == (640, 0)
    assert white_region(9980, 640, True) == (0, 640)
    assert white_region(-9980, 640, True) == (0, 0)


def test_white_sits_at_the_bottom_unless_the_board_is_flipped():
    top, size = white_region(300, 640, False)
    assert top + size == 640
    top, size = white_region(300, 640, True)
    assert top == 0


def test_the_label_sits_at_the_end_of_the_side_that_is_ahead():
    assert not label_at_top(120, False)
    assert label_at_top(-120, False)
    assert label_at_top(120, True)
    assert not label_at_top(-120, True)
    assert not label_at_top(0, False)


@pytest.mark.parametrize(
    "score, expected",
    [(0, "0.0"), (6, "0.1"), (-123, "1.2"), (1234, "12.3"), (9980, "M2"), (-9990, "M1"), (10000, "#"), (-10000, "#")],
)
def test_the_label_gives_the_size_of_the_advantage_without_its_sign(score, expected):
    assert bar_label(score) == expected