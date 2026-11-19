import pytest

from chess.board import Board
from chess.notation import parse_uci
from chess.pgn import uci_moves_to_san
from chess.rules import generate_legal_moves
from chess.san import move_to_san

CASTLING = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


def san(fen: str, uci: str) -> str:
    board = Board.from_fen(fen)
    move = parse_uci(uci, generate_legal_moves(board, board.turn))
    return move_to_san(board, move)


@pytest.mark.parametrize(
    "fen, uci, expected",
    [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", "e4"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3", "Nf3"),
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", "exd5"),
        ("4k3/8/8/4p3/8/5N2/8/4K3 w - - 0 1", "f3e5", "Nxe5"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", "exd6"),
        (CASTLING, "e1g1", "O-O"),
        (CASTLING, "e1c1", "O-O-O"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", "a8=N"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", "a8=Q+"),
        ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8n", "axb8=N"),
        ("4k3/8/8/8/8/8/8/4K2R w - - 0 1", "h1h8", "Rh8+"),
    ],
)
def test_basic_notation(fen, uci, expected):
    assert san(fen, uci) == expected


@pytest.mark.parametrize(
    "fen, uci, expected",
    [
        ("4k3/8/8/8/8/8/8/N1N1K3 w - - 0 1", "a1b3", "Nab3"),
        ("4k3/8/8/8/8/8/8/N1N1K3 w - - 0 1", "c1b3", "Ncb3"),
        ("4k3/8/8/R7/8/8/8/R3K3 w - - 0 1", "a5a3", "R5a3"),
        ("4k3/8/8/R7/8/8/8/R3K3 w - - 0 1", "a1a3", "R1a3"),
        ("8/7k/8/Q7/8/8/8/Q3Q1K1 w - - 0 1", "a1c3", "Qa1c3"),
        ("8/7k/8/Q7/8/8/8/Q3Q1K1 w - - 0 1", "e1c3", "Qec3"),
        ("8/7k/8/Q7/8/8/8/Q3Q1K1 w - - 0 1", "a5c3", "Q5c3"),
    ],
)
def test_disambiguation(fen, uci, expected):
    assert san(fen, uci) == expected


def test_san_leaves_the_board_untouched():
    board = Board.from_fen(CASTLING)
    before = board.position_key()
    move = parse_uci("e1g1", generate_legal_moves(board, board.turn))
    move_to_san(board, move)
    assert board.position_key() == before
    assert board.history == []


def test_missing_piece_is_rejected():
    from chess.move import Move

    board = Board.starting_position()
    with pytest.raises(ValueError):
        move_to_san(board, Move((4, 4), (3, 4)))


def test_scholars_mate():
    moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
    assert uci_moves_to_san(moves) == ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]


def test_fools_mate():
    assert uci_moves_to_san(["f2f3", "e7e5", "g2g4", "d8h4"]) == ["f3", "e5", "g4", "Qh4#"]


def test_illegal_uci_move_is_rejected():
    with pytest.raises(ValueError):
        uci_moves_to_san(["e2e5"])