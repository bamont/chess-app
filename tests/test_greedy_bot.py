import pytest

from bot.greedy_bot import GreedyBot
from chess.board import Board
from chess.notation import move_to_uci
from chess.rules import generate_legal_moves


def test_returns_a_legal_move():
    board = Board.starting_position()
    assert GreedyBot(seed=0).choose_move(board) in generate_legal_moves(board, board.turn)


def test_captures_a_free_queen():
    board = Board.from_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    assert move_to_uci(GreedyBot(seed=0).choose_move(board)) == "e4d5"


def test_falls_for_a_defended_pawn():
    board = Board.from_fen("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1")
    assert move_to_uci(GreedyBot(seed=0).choose_move(board)) == "d1d5"


def test_does_not_modify_the_board():
    board = Board.starting_position()
    before = board.position_key()
    GreedyBot(seed=0).choose_move(board)
    assert board.position_key() == before
    assert board.history == []


def test_raises_when_no_legal_move():
    with pytest.raises(ValueError):
        GreedyBot(seed=0).choose_move(Board.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))