import random

import pytest

from bot.evaluation import PIECE_VALUES, evaluate_positional, piece_square_value
from chess.board import Board
from chess.pieces import Color, Kind
from chess.rules import generate_legal_moves, has_insufficient_material

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/1N2K3 b - - 0 1",
    "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",
    "2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
    "4k1n1/8/8/8/8/8/8/4K1N1 w - - 0 1",
    "4k3/8/8/8/8/8/8/3NKN2 w - - 0 1",
    "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "3qk3/8/8/8/8/8/8/2BQK3 w - - 0 1",
    "3qk3/8/8/8/8/8/8/1NBQK3 b - - 0 1",
    "3qk3/8/8/8/8/8/8/R2QK3 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "8/8/8/8/8/8/8/8 w - - 0 1",
    "8/8/8/8/8/8/8/4K3 w - - 0 1",
]


def reference_is_endgame(board: Board) -> bool:
    queens = {Color.WHITE: 0, Color.BLACK: 0}
    rooks = {Color.WHITE: 0, Color.BLACK: 0}
    minors = {Color.WHITE: 0, Color.BLACK: 0}
    for row in board.grid:
        for piece in row:
            if piece is None:
                continue
            if piece.kind == Kind.QUEEN:
                queens[piece.color] += 1
            elif piece.kind == Kind.ROOK:
                rooks[piece.color] += 1
            elif piece.kind in (Kind.BISHOP, Kind.KNIGHT):
                minors[piece.color] += 1
    return not any(queens[c] > 0 and (rooks[c] > 0 or minors[c] > 1) for c in Color)


def reference_positional(board: Board) -> int:
    endgame = reference_is_endgame(board)
    score = 0
    for row in range(8):
        for col in range(8):
            piece = board.grid[row][col]
            if piece is None:
                continue
            value = PIECE_VALUES[piece.kind] + piece_square_value(piece, row, col, endgame)
            score += value if piece.color == board.turn else -value
    return score


def reference_insufficient_material(board: Board) -> bool:
    others = []
    for row in range(8):
        for col in range(8):
            piece = board.grid[row][col]
            if piece is not None and piece.kind != Kind.KING:
                others.append((row, col, piece))
    if not others:
        return True
    if len(others) == 1:
        return others[0][2].kind in (Kind.BISHOP, Kind.KNIGHT)
    if all(piece.kind == Kind.BISHOP for _, _, piece in others):
        return len({(row + col) % 2 for row, col, _ in others}) == 1
    return False


@pytest.mark.parametrize("fen", POSITIONS)
def test_positional_evaluation_matches_the_reference(fen):
    board = Board.from_fen(fen)
    assert evaluate_positional(board) == reference_positional(board)


@pytest.mark.parametrize("fen", POSITIONS)
def test_insufficient_material_matches_the_reference(fen):
    board = Board.from_fen(fen)
    assert has_insufficient_material(board) == reference_insufficient_material(board)


@pytest.mark.parametrize("seed", range(5))
def test_both_match_the_reference_along_random_games(seed):
    rng = random.Random(seed)
    board = Board.starting_position()
    for _ in range(100):
        assert evaluate_positional(board) == reference_positional(board)
        assert has_insufficient_material(board) == reference_insufficient_material(board)
        moves = generate_legal_moves(board, board.turn)
        if not moves:
            break
        board.make_move(rng.choice(moves))