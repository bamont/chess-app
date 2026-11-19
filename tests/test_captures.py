import pytest

from chess.board import Board
from chess.rules import generate_legal_captures, generate_legal_moves

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
    "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
]


@pytest.mark.parametrize("fen", POSITIONS)
def test_captures_are_the_capturing_subset_of_legal_moves(fen):
    board = Board.from_fen(fen)
    expected = {
        move
        for move in generate_legal_moves(board, board.turn)
        if move.captured is not None or move.promotion is not None
    }
    assert set(generate_legal_captures(board, board.turn)) == expected


def test_generation_leaves_the_board_untouched():
    board = Board.from_fen(POSITIONS[1])
    before = board.position_key()
    generate_legal_captures(board, board.turn)
    assert board.position_key() == before
    assert board.history == []