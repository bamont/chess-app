import pytest

from chess.board import Board
from chess.move import Move
from chess.pieces import Color, Kind, Piece
from chess.utils import to_square

WHITE = Color.WHITE
BLACK = Color.BLACK


def snapshot(board: Board):
    return (
        [row[:] for row in board.grid],
        board.turn,
        dict(board.castling_rights),
        board.en_passant,
        board.halfmove_clock,
    )


class TestFen:
    def test_start_position(self):
        board = Board.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        expected = Board.starting_position()
        assert snapshot(board) == snapshot(expected)

    def test_black_to_move_with_en_passant_square(self):
        board = Board.from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        assert board.turn == BLACK
        assert board.en_passant == to_square("e3")
        assert board.piece_at(to_square("e4")) == Piece(WHITE, Kind.PAWN)

    def test_partial_castling_rights(self):
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")
        assert board.castling_rights == {"wK": True, "wQ": False, "bK": False, "bQ": True}

    def test_no_castling_rights_and_halfmove_clock(self):
        board = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 37 60")
        assert not any(board.castling_rights.values())
        assert board.halfmove_clock == 37

    def test_invalid_fen_raises(self):
        with pytest.raises(ValueError):
            Board.from_fen("not a fen")


class TestMakeMoveBookkeeping:
    def test_double_push_sets_en_passant_square(self):
        board = Board.starting_position()
        board.make_move(Move(to_square("e2"), to_square("e4")))
        assert board.en_passant == to_square("e3")
        board.make_move(Move(to_square("e7"), to_square("e5")))
        assert board.en_passant == to_square("e6")
        board.make_move(Move(to_square("g1"), to_square("f3")))
        assert board.en_passant is None

    def test_halfmove_clock_resets_on_pawn_move_and_increments_otherwise(self):
        board = Board.starting_position()
        board.make_move(Move(to_square("g1"), to_square("f3")))
        assert board.halfmove_clock == 1
        board.make_move(Move(to_square("e7"), to_square("e5")))
        assert board.halfmove_clock == 0

    def test_halfmove_clock_resets_on_capture(self):
        board = Board.from_fen("4k3/8/8/8/8/8/p7/R3K3 w - - 7 1")
        board.make_move(Move(to_square("a1"), to_square("a2"), captured=Piece(BLACK, Kind.PAWN)))
        assert board.halfmove_clock == 0

    def test_king_move_removes_both_castling_rights(self):
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        board.make_move(Move(to_square("e1"), to_square("e2")))
        assert board.castling_rights == {"wK": False, "wQ": False, "bK": True, "bQ": True}

    def test_rook_move_removes_matching_right(self):
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        board.make_move(Move(to_square("h1"), to_square("h2")))
        assert board.castling_rights == {"wK": False, "wQ": True, "bK": True, "bQ": True}

    def test_capturing_rook_on_corner_removes_opponent_right(self):
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        board.make_move(Move(to_square("a1"), to_square("a8"), captured=Piece(BLACK, Kind.ROOK)))
        assert board.castling_rights == {"wK": True, "wQ": False, "bK": True, "bQ": False}


class TestUndoRestoresFullState:
    @pytest.mark.parametrize(
        "fen, start, end",
        [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2", "e4"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 10", "a1", "a2"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 10", "e1", "e2"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 10", "h1", "h8"),
        ],
    )
    def test_undo(self, fen, start, end):
        board = Board.from_fen(fen)
        before = snapshot(board)
        s, e = to_square(start), to_square(end)
        board.make_move(Move(s, e, captured=board.piece_at(e)))
        board.undo_move()
        assert snapshot(board) == before
        assert board.history == []


class TestPromotion:
    def test_white_promotion_places_chosen_piece(self):
        board = Board.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        board.make_move(Move(to_square("a7"), to_square("a8"), promotion=Kind.QUEEN))
        assert board.piece_at(to_square("a8")) == Piece(WHITE, Kind.QUEEN)
        assert board.piece_at(to_square("a7")) is None

    def test_black_promotion_to_knight(self):
        board = Board.from_fen("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
        board.make_move(Move(to_square("a2"), to_square("a1"), promotion=Kind.KNIGHT))
        assert board.piece_at(to_square("a1")) == Piece(BLACK, Kind.KNIGHT)

    def test_undo_promotion_restores_pawn(self):
        board = Board.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        before = snapshot(board)
        board.make_move(Move(to_square("a7"), to_square("a8"), promotion=Kind.QUEEN))
        board.undo_move()
        assert snapshot(board) == before
        assert board.piece_at(to_square("a7")) == Piece(WHITE, Kind.PAWN)

    def test_undo_capture_promotion_restores_captured_piece(self):
        board = Board.from_fen("1r2k3/P7/8/8/8/8/8/4K3 w - - 3 1")
        before = snapshot(board)
        rook = Piece(BLACK, Kind.ROOK)
        board.make_move(Move(to_square("a7"), to_square("b8"), captured=rook, promotion=Kind.QUEEN))
        assert board.piece_at(to_square("b8")) == Piece(WHITE, Kind.QUEEN)
        assert board.halfmove_clock == 0
        board.undo_move()
        assert snapshot(board) == before

    def test_capture_promotion_on_corner_removes_castling_right(self):
        board = Board.from_fen("4k2r/6P1/8/8/8/8/8/4K3 w k - 0 1")
        rook = Piece(BLACK, Kind.ROOK)
        board.make_move(Move(to_square("g7"), to_square("h8"), captured=rook, promotion=Kind.QUEEN))
        assert board.castling_rights["bK"] is False


class TestEnPassantMake:
    def test_white_en_passant_removes_captured_pawn(self):
        board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        pawn = Piece(BLACK, Kind.PAWN)
        board.make_move(Move(to_square("e5"), to_square("d6"), captured=pawn, flag="en_passant"))
        assert board.piece_at(to_square("d6")) == Piece(WHITE, Kind.PAWN)
        assert board.piece_at(to_square("d5")) is None
        assert board.piece_at(to_square("e5")) is None
        assert board.en_passant is None
        assert board.halfmove_clock == 0

    def test_black_en_passant_removes_captured_pawn(self):
        board = Board.from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
        pawn = Piece(WHITE, Kind.PAWN)
        board.make_move(Move(to_square("d4"), to_square("e3"), captured=pawn, flag="en_passant"))
        assert board.piece_at(to_square("e3")) == Piece(BLACK, Kind.PAWN)
        assert board.piece_at(to_square("e4")) is None

    def test_undo_restores_both_pawns(self):
        board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 4 1")
        before = snapshot(board)
        pawn = Piece(BLACK, Kind.PAWN)
        board.make_move(Move(to_square("e5"), to_square("d6"), captured=pawn, flag="en_passant"))
        board.undo_move()
        assert snapshot(board) == before
        assert board.piece_at(to_square("d5")) == pawn
        assert board.piece_at(to_square("d6")) is None


CASTLING_CASES = [
    ("w", WHITE, "e1", "g1", "h1", "f1", "castle_k"),
    ("w", WHITE, "e1", "c1", "a1", "d1", "castle_q"),
    ("b", BLACK, "e8", "g8", "h8", "f8", "castle_k"),
    ("b", BLACK, "e8", "c8", "a8", "d8", "castle_q"),
]


@pytest.mark.parametrize("side, color, king_start, king_end, rook_start, rook_end, flag", CASTLING_CASES)
def test_castling_moves_king_and_rook(side, color, king_start, king_end, rook_start, rook_end, flag):
    board = Board.from_fen(f"r3k2r/8/8/8/8/8/8/R3K2R {side} KQkq - 3 1")
    board.make_move(Move(to_square(king_start), to_square(king_end), flag=flag))
    assert board.piece_at(to_square(king_end)) == Piece(color, Kind.KING)
    assert board.piece_at(to_square(rook_end)) == Piece(color, Kind.ROOK)
    assert board.piece_at(to_square(king_start)) is None
    assert board.piece_at(to_square(rook_start)) is None
    own = color.value
    other = color.opponent().value
    assert not board.castling_rights[f"{own}K"]
    assert not board.castling_rights[f"{own}Q"]
    assert board.castling_rights[f"{other}K"]
    assert board.castling_rights[f"{other}Q"]
    assert board.halfmove_clock == 4
    assert board.turn == color.opponent()


@pytest.mark.parametrize("side, color, king_start, king_end, rook_start, rook_end, flag", CASTLING_CASES)
def test_undo_castling_restores_state(side, color, king_start, king_end, rook_start, rook_end, flag):
    board = Board.from_fen(f"r3k2r/8/8/8/8/8/8/R3K2R {side} KQkq - 3 1")
    before = snapshot(board)
    board.make_move(Move(to_square(king_start), to_square(king_end), flag=flag))
    board.undo_move()
    assert snapshot(board) == before


class TestPositionKey:
    def test_same_position_same_key(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert Board.starting_position().position_key() == Board.from_fen(fen).position_key()

    def test_side_to_move_matters(self):
        white = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        black = Board.from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
        assert white.position_key() != black.position_key()

    def test_castling_rights_matter(self):
        with_rights = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        without = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
        assert with_rights.position_key() != without.position_key()

    def test_key_is_hashable(self):
        assert isinstance(hash(Board.starting_position().position_key()), int)


class TestCopy:
    def test_copy_has_the_same_position(self):
        board = Board.starting_position()
        assert board.copy().position_key() == board.position_key()

    def test_copy_is_independent(self):
        board = Board.starting_position()
        clone = board.copy()
        clone.make_move(Move(to_square("e2"), to_square("e4")))
        assert board.position_key() == Board.starting_position().position_key()
        assert board.history == []

    def test_castling_rights_are_independent(self):
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        clone = board.copy()
        clone.make_move(Move(to_square("e1"), to_square("e2")))
        assert board.castling_rights["wK"] is True

    def test_copy_keeps_the_history(self):
        board = Board.starting_position()
        board.make_move(Move(to_square("e2"), to_square("e4")))
        board.make_move(Move(to_square("e7"), to_square("e5")))
        clone = board.copy()
        assert len(clone.history) == 2
        assert clone.position_key() == board.position_key()
        assert clone.halfmove_clock == board.halfmove_clock

    def test_undoing_on_a_copy_leaves_the_original_untouched(self):
        board = Board.starting_position()
        board.make_move(Move(to_square("e2"), to_square("e4")))
        before = board.position_key()
        clone = board.copy()
        clone.undo_move()
        assert clone.position_key() == Board.starting_position().position_key()
        assert board.position_key() == before
        assert len(board.history) == 1

    def test_playing_after_undoing_on_a_copy_does_not_corrupt_the_original_history(self):
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        board.make_move(Move(to_square("e1"), to_square("e2")))
        clone = board.copy()
        clone.undo_move()
        clone.make_move(Move(to_square("h1"), to_square("h2")))
        board.undo_move()
        assert board.castling_rights == {"wK": True, "wQ": True, "bK": True, "bQ": True}