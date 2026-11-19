import pytest

pytest.importorskip("numpy")

from motifs.audit import game_fens  # noqa: E402


def test_positions_are_listed_from_the_first_wanted_ply():
    fens = game_fens(["e4", "e5", "Nf3"], 1)
    assert len(fens) == 2
    assert fens[0].startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b")
    assert fens[1].startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w")


def test_no_position_is_listed_before_the_wanted_ply():
    assert game_fens(["e4", "e5"], 5) == []
    assert len(game_fens(["e4", "e5"], 0)) == 2