import pytest

from chess.move import Move
from chess.pieces import Color, Kind, Piece


def test_enum_values_are_unchanged():
    assert Color.WHITE.value == "w"
    assert Color.BLACK.value == "b"
    assert [kind.value for kind in Kind] == ["P", "N", "B", "R", "Q", "K"]
    assert Kind("Q") is Kind.QUEEN
    assert Color("w") is Color.WHITE


def test_opponent():
    assert Color.WHITE.opponent() is Color.BLACK
    assert Color.BLACK.opponent() is Color.WHITE


def test_piece_equality_and_hashing():
    pawn = Piece(Color.WHITE, Kind.PAWN)
    assert pawn == Piece(Color.WHITE, Kind.PAWN)
    assert pawn != Piece(Color.BLACK, Kind.PAWN)
    assert len({pawn, Piece(Color.WHITE, Kind.PAWN)}) == 1
    assert pawn.color is Color.WHITE
    assert pawn.kind is Kind.PAWN


def test_move_defaults_and_equality():
    move = Move((6, 4), (4, 4))
    assert move.captured is None
    assert move.promotion is None
    assert move.flag is None
    assert move == Move((6, 4), (4, 4))
    assert move != Move((6, 4), (4, 4), flag="en_passant")
    assert hash(move) == hash(Move((6, 4), (4, 4)))


def test_move_with_keywords():
    move = Move((1, 0), (0, 0), captured=Piece(Color.BLACK, Kind.ROOK), promotion=Kind.QUEEN)
    assert move.promotion is Kind.QUEEN
    assert move.captured.kind is Kind.ROOK


def test_pieces_and_moves_are_immutable():
    with pytest.raises(AttributeError):
        Piece(Color.WHITE, Kind.PAWN).kind = Kind.QUEEN
    with pytest.raises(AttributeError):
        Move((6, 4), (4, 4)).flag = "castle_k"