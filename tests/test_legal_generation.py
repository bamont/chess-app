import random

import pytest

from chess.board import Board
from chess.rules import (
    find_king,
    generate_legal_captures,
    generate_legal_moves,
    generate_pseudo_legal_moves,
    is_in_check,
    pinned_squares,
)
from chess.utils import from_square

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "8/8/8/8/k2Pp2Q/8/8/4K3 b - d3 0 1",
    "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
]


def brute_force_legal_moves(board: Board, color):
    legal = []
    for move in generate_pseudo_legal_moves(board, color):
        board.make_move(move)
        if not is_in_check(board, color):
            legal.append(move)
        board.undo_move()
    return legal


def pinned(fen: str) -> set[str]:
    board = Board.from_fen(fen)
    squares = pinned_squares(board, board.turn, find_king(board, board.turn))
    return {from_square(square) for square in squares}


@pytest.mark.parametrize("fen", POSITIONS)
def test_legal_moves_match_brute_force_on_reference_positions(fen):
    board = Board.from_fen(fen)
    fast = generate_legal_moves(board, board.turn)
    slow = brute_force_legal_moves(board, board.turn)
    assert len(fast) == len(slow)
    assert set(fast) == set(slow)


@pytest.mark.parametrize("fen", POSITIONS)
def test_legal_captures_match_brute_force_on_reference_positions(fen):
    board = Board.from_fen(fen)
    expected = {
        move
        for move in brute_force_legal_moves(board, board.turn)
        if move.captured is not None or move.promotion is not None
    }
    assert set(generate_legal_captures(board, board.turn)) == expected


@pytest.mark.parametrize("seed", range(5))
def test_legal_moves_match_brute_force_along_random_games(seed):
    rng = random.Random(seed)
    board = Board.starting_position()
    for _ in range(80):
        fast = generate_legal_moves(board, board.turn)
        slow = brute_force_legal_moves(board, board.turn)
        assert len(fast) == len(slow)
        assert set(fast) == set(slow)
        if not fast:
            break
        board.make_move(rng.choice(fast))


@pytest.mark.parametrize("fen", POSITIONS)
def test_generation_leaves_the_board_untouched(fen):
    board = Board.from_fen(fen)
    before = board.position_key()
    generate_legal_moves(board, board.turn)
    generate_legal_captures(board, board.turn)
    assert board.position_key() == before
    assert board.history == []


def test_piece_pinned_on_a_file():
    assert pinned("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1") == {"e2"}


def test_piece_pinned_on_a_diagonal():
    assert pinned("4k3/8/8/b7/8/8/3B4/4K3 w - - 0 1") == {"d2"}


def test_piece_pinned_on_a_rank_by_a_queen():
    assert pinned("4k3/8/8/8/8/8/8/q1B1K3 w - - 0 1") == {"c1"}


def test_two_blockers_mean_no_pin():
    assert pinned("4r1k1/8/8/8/8/4N3/4R3/4K3 w - - 0 1") == set()


def test_wrong_kind_of_slider_does_not_pin():
    assert pinned("4b1k1/8/8/8/8/8/4R3/4K3 w - - 0 1") == set()


def test_enemy_piece_in_between_means_no_pin():
    assert pinned("4r1k1/8/8/8/8/4p3/4R3/4K3 w - - 0 1") == set()