import pytest

from bot.minimax_bot import MinimaxBot
from chess.board import Board
from chess.notation import move_to_uci
from chess.rules import generate_legal_moves


def test_returns_a_legal_move():
    board = Board.starting_position()
    assert MinimaxBot(depth=2, seed=0).choose_move(board) in generate_legal_moves(board, board.turn)


def test_invalid_depth_is_rejected():
    with pytest.raises(ValueError):
        MinimaxBot(depth=0)


@pytest.mark.parametrize("depth", [1, 2])
def test_finds_mate_in_one(depth):
    board = Board.from_fen("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    assert move_to_uci(MinimaxBot(depth=depth, seed=0).choose_move(board)) == "a1a8"


def test_captures_a_free_queen():
    board = Board.from_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    assert move_to_uci(MinimaxBot(depth=2, seed=0).choose_move(board)) == "e4d5"


def test_does_not_fall_for_a_defended_pawn():
    board = Board.from_fen("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1")
    assert move_to_uci(MinimaxBot(depth=2, seed=0).choose_move(board)) != "d1d5"


def test_same_seed_gives_the_same_move():
    board = Board.starting_position()
    first = MinimaxBot(depth=2, seed=3).choose_move(board)
    second = MinimaxBot(depth=2, seed=3).choose_move(board)
    assert first == second


def test_does_not_modify_the_board():
    board = Board.starting_position()
    before = board.position_key()
    MinimaxBot(depth=2, seed=0).choose_move(board)
    assert board.position_key() == before
    assert board.history == []