import pytest

from bot.evaluation import evaluate_positional, is_endgame
from chess.board import Board

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.mark.parametrize(
    "fen",
    [
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "3qk3/8/8/8/8/8/8/3QK3 w - - 0 1",
        "3qk3/8/8/8/8/8/8/2BQK3 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1",
    ],
)
def test_endgame_positions(fen):
    assert is_endgame(Board.from_fen(fen))


@pytest.mark.parametrize(
    "fen",
    [
        START,
        "3qk3/8/8/8/8/8/8/R2QK3 w - - 0 1",
        "3qk3/8/8/8/8/8/8/1NBQK3 w - - 0 1",
    ],
)
def test_middlegame_positions(fen):
    assert not is_endgame(Board.from_fen(fen))


def test_start_position_is_balanced():
    assert evaluate_positional(Board.from_fen(START)) == 0


def test_advanced_pawn_is_worth_more():
    assert evaluate_positional(Board.from_fen("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")) == 120
    assert evaluate_positional(Board.from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")) == 80


def test_tables_are_mirrored_for_black():
    assert evaluate_positional(Board.from_fen("4k3/8/8/4p3/8/8/8/4K3 b - - 0 1")) == 120


def test_centralized_knight_beats_rim_knight():
    assert evaluate_positional(Board.from_fen("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1")) == 330
    assert evaluate_positional(Board.from_fen("4k3/8/8/8/8/7N/8/4K3 w - - 0 1")) == 290


def test_king_is_centralized_in_the_endgame():
    assert evaluate_positional(Board.from_fen("4k3/8/8/8/4K3/8/8/8 w - - 0 1")) == 70
    assert evaluate_positional(Board.from_fen("4k3/8/8/8/8/8/8/K7 w - - 0 1")) == -20


def test_score_flips_with_the_side_to_move():
    white = Board.from_fen("4k3/8/8/8/4P3/5N2/8/4K3 w - - 0 1")
    black = Board.from_fen("4k3/8/8/8/4P3/5N2/8/4K3 b - - 0 1")
    assert evaluate_positional(black) == -evaluate_positional(white)