import random

import pytest

from chess.board import Board
from chess.pieces import Color
from chess.rules import (
    BISHOP_RAYS,
    KING_TARGETS,
    KNIGHT_TARGETS,
    ROOK_RAYS,
    generate_legal_moves,
    generate_pseudo_legal_captures,
    generate_pseudo_legal_moves,
)

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
    "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
]


def capturing(moves):
    return [move for move in moves if move.captured is not None or move.promotion is not None]


def count(table, squares=range(8)):
    return sum(len(entry) for row in squares for entry in table[row])


def test_knight_targets():
    assert len(KNIGHT_TARGETS[0][0]) == 2
    assert len(KNIGHT_TARGETS[4][4]) == 8
    assert count(KNIGHT_TARGETS) == 336


def test_king_targets():
    assert len(KING_TARGETS[0][0]) == 3
    assert len(KING_TARGETS[0][4]) == 5
    assert len(KING_TARGETS[4][4]) == 8
    assert count(KING_TARGETS) == 420


def test_rook_rays():
    assert all(sum(len(ray) for ray in ROOK_RAYS[r][c]) == 14 for r in range(8) for c in range(8))
    assert ROOK_RAYS[4][4][0] == ((4, 5), (4, 6), (4, 7))
    assert sum(len(ray) for r in range(8) for c in range(8) for ray in ROOK_RAYS[r][c]) == 896


def test_bishop_rays():
    assert sum(len(ray) for ray in BISHOP_RAYS[0][0]) == 7
    assert sum(len(ray) for ray in BISHOP_RAYS[3][3]) == 13
    assert sum(len(ray) for r in range(8) for c in range(8) for ray in BISHOP_RAYS[r][c]) == 560


@pytest.mark.parametrize("fen", POSITIONS)
@pytest.mark.parametrize("color", list(Color))
def test_pseudo_legal_captures_are_the_capturing_pseudo_legal_moves(fen, color):
    board = Board.from_fen(fen)
    assert generate_pseudo_legal_captures(board, color) == capturing(generate_pseudo_legal_moves(board, color))


@pytest.mark.parametrize("seed", range(5))
def test_pseudo_legal_captures_along_random_games(seed):
    rng = random.Random(seed)
    board = Board.starting_position()
    for _ in range(80):
        for color in Color:
            assert generate_pseudo_legal_captures(board, color) == capturing(
                generate_pseudo_legal_moves(board, color)
            )
        moves = generate_legal_moves(board, board.turn)
        if not moves:
            break
        board.make_move(rng.choice(moves))