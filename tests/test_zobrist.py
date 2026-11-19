import random

import pytest

from chess.board import Board
from chess.notation import parse_uci
from chess.rules import generate_legal_moves

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CASTLING = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

SPECIAL_CASES = [
    (CASTLING, ["e1g1", "e8c8"]),
    (CASTLING, ["e1c1", "e8g8"]),
    (CASTLING, ["a1a8", "e8e7"]),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"]),
    ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7a8q"]),
    ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7b8n"]),
    ("4k3/8/8/8/8/8/4p3/K7 b - - 0 1", ["e2e1q"]),
]


def play(board: Board, *ucis: str) -> None:
    for uci in ucis:
        board.make_move(parse_uci(uci, generate_legal_moves(board, board.turn)))


def test_starting_position_matches_its_fen():
    assert Board.starting_position().zobrist == Board.from_fen(START).zobrist
    assert Board.starting_position().zobrist == Board.starting_position().compute_zobrist()


@pytest.mark.parametrize("seed", range(5))
def test_incremental_hash_matches_full_computation_along_random_games(seed):
    rng = random.Random(seed)
    board = Board.starting_position()
    hashes = [board.zobrist]
    for _ in range(100):
        moves = generate_legal_moves(board, board.turn)
        if not moves:
            break
        board.make_move(rng.choice(moves))
        assert board.zobrist == board.compute_zobrist()
        hashes.append(board.zobrist)
    while board.history:
        board.undo_move()
        hashes.pop()
        assert board.zobrist == hashes[-1]
        assert board.zobrist == board.compute_zobrist()


@pytest.mark.parametrize("fen, ucis", SPECIAL_CASES)
def test_special_moves_keep_the_hash_consistent(fen, ucis):
    board = Board.from_fen(fen)
    start_hash = board.zobrist
    for uci in ucis:
        play(board, uci)
        assert board.zobrist == board.compute_zobrist()
    for _ in ucis:
        board.undo_move()
    assert board.zobrist == start_hash


def test_transpositions_share_the_same_hash():
    first, second = Board.starting_position(), Board.starting_position()
    play(first, "g1f3", "g8f6", "b1c3", "b8c6")
    play(second, "b1c3", "b8c6", "g1f3", "g8f6")
    assert first.zobrist == second.zobrist


def test_hash_depends_on_the_side_to_move():
    white = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    black = Board.from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert white.zobrist != black.zobrist


def test_hash_depends_on_castling_rights():
    with_rights = Board.from_fen(CASTLING)
    without = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert with_rights.zobrist != without.zobrist


def test_hash_depends_on_the_en_passant_file():
    with_square = Board.from_fen("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1")
    without = Board.from_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1")
    assert with_square.zobrist != without.zobrist


def test_different_positions_have_different_hashes():
    board = Board.starting_position()
    before = board.zobrist
    play(board, "e2e4")
    assert board.zobrist != before


def test_copy_keeps_the_hash_and_the_hashes_of_the_history():
    board = Board.starting_position()
    play(board, "e2e4", "e7e5")
    clone = board.copy()
    assert clone.zobrist == board.zobrist
    clone.undo_move()
    clone.undo_move()
    assert clone.zobrist == Board.starting_position().zobrist
    assert board.zobrist == board.compute_zobrist()


def test_recompute_repairs_a_hand_edited_board():
    board = Board.starting_position()
    board.grid[6][4] = None
    assert board.zobrist != board.compute_zobrist()
    board.recompute_zobrist()
    assert board.zobrist == board.compute_zobrist()