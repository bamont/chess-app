import random

import pytest

from chess.pieces import Color
from ui.options import resolve_color


def test_fixed_choices():
    assert resolve_color("white") == Color.WHITE
    assert resolve_color("black") == Color.BLACK


def test_random_choice_gives_both_colors():
    rng = random.Random(0)
    assert {resolve_color("random", rng) for _ in range(50)} == {Color.WHITE, Color.BLACK}


def test_random_choice_is_reproducible_with_a_seed():
    first, second = random.Random(5), random.Random(5)
    assert [resolve_color("random", first) for _ in range(10)] == [resolve_color("random", second) for _ in range(10)]


def test_unknown_choice_raises():
    with pytest.raises(ValueError):
        resolve_color("purple")