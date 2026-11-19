import pytest

from chess.board import Board
from chess.move import Move
from chess.pieces import Color, Kind, Piece
from chess.rules import (
    bishop_moves,
    generate_legal_moves,
    generate_pseudo_legal_moves,
    is_checkmate,
    is_in_check,
    is_square_attacked,
    is_stalemate,
    king_moves,
    knight_moves,
    pawn_moves,
    queen_moves,
    rook_moves,
)
from chess.utils import from_square, to_square

WHITE = Color.WHITE
BLACK = Color.BLACK
EXPECTED_PROMOTIONS = {Kind.QUEEN, Kind.ROOK, Kind.BISHOP, Kind.KNIGHT}


def make_board(*placements: tuple[str, Color, Kind]) -> Board:
    board = Board()
    for notation, color, kind in placements:
        row, col = to_square(notation)
        board.grid[row][col] = Piece(color, kind)
    return board


def destinations(moves: list[Move]) -> set[str]:
    return {from_square(move.end) for move in moves}


def find_move(moves: list[Move], end: str) -> Move:
    return next(move for move in moves if from_square(move.end) == end)


class TestKnight:
    def test_start_position(self):
        board = Board.starting_position()
        assert destinations(knight_moves(board, to_square("b1"))) == {"a3", "c3"}

    def test_center_has_eight_moves(self):
        board = make_board(("e4", WHITE, Kind.KNIGHT))
        assert len(knight_moves(board, to_square("e4"))) == 8

    def test_corner(self):
        board = make_board(("a1", WHITE, Kind.KNIGHT))
        assert destinations(knight_moves(board, to_square("a1"))) == {"b3", "c2"}

    def test_cannot_capture_ally(self):
        board = make_board(("e4", WHITE, Kind.KNIGHT), ("d6", WHITE, Kind.PAWN))
        moves = knight_moves(board, to_square("e4"))
        assert len(moves) == 7
        assert "d6" not in destinations(moves)

    def test_captures_enemy(self):
        board = make_board(("e4", WHITE, Kind.KNIGHT), ("d6", BLACK, Kind.PAWN))
        moves = knight_moves(board, to_square("e4"))
        assert len(moves) == 8
        assert find_move(moves, "d6").captured == Piece(BLACK, Kind.PAWN)

    def test_normal_move_has_no_capture(self):
        board = make_board(("e4", WHITE, Kind.KNIGHT))
        assert all(move.captured is None for move in knight_moves(board, to_square("e4")))


class TestKing:
    def test_start_position_has_no_moves(self):
        board = Board.starting_position()
        assert king_moves(board, to_square("e1")) == []

    def test_center_has_eight_moves(self):
        board = make_board(("e4", WHITE, Kind.KING))
        assert len(king_moves(board, to_square("e4"))) == 8

    def test_corner(self):
        board = make_board(("a1", WHITE, Kind.KING))
        assert destinations(king_moves(board, to_square("a1"))) == {"a2", "b1", "b2"}

    def test_blocked_by_ally_and_captures_enemy(self):
        board = make_board(
            ("e4", WHITE, Kind.KING),
            ("e5", WHITE, Kind.PAWN),
            ("d5", BLACK, Kind.PAWN),
        )
        moves = king_moves(board, to_square("e4"))
        assert len(moves) == 7
        assert "e5" not in destinations(moves)
        assert find_move(moves, "d5").captured == Piece(BLACK, Kind.PAWN)


class TestRook:
    def test_empty_board(self):
        board = make_board(("e4", WHITE, Kind.ROOK))
        assert len(rook_moves(board, to_square("e4"))) == 14

    def test_start_position_has_no_moves(self):
        board = Board.starting_position()
        assert rook_moves(board, to_square("a1")) == []

    def test_blocked_by_ally(self):
        board = make_board(("e4", WHITE, Kind.ROOK), ("g4", WHITE, Kind.PAWN))
        moves = rook_moves(board, to_square("e4"))
        assert len(moves) == 12
        assert {"g4", "h4"}.isdisjoint(destinations(moves))
        assert "f4" in destinations(moves)

    def test_stops_on_enemy_capture(self):
        board = make_board(("e4", WHITE, Kind.ROOK), ("g4", BLACK, Kind.PAWN))
        moves = rook_moves(board, to_square("e4"))
        assert len(moves) == 13
        assert "h4" not in destinations(moves)
        assert find_move(moves, "g4").captured == Piece(BLACK, Kind.PAWN)


class TestBishop:
    def test_empty_board(self):
        board = make_board(("e4", WHITE, Kind.BISHOP))
        assert len(bishop_moves(board, to_square("e4"))) == 13

    def test_start_position_has_no_moves(self):
        board = Board.starting_position()
        assert bishop_moves(board, to_square("c1")) == []

    def test_blocked_by_ally(self):
        board = make_board(("e4", WHITE, Kind.BISHOP), ("g6", WHITE, Kind.PAWN))
        moves = bishop_moves(board, to_square("e4"))
        assert len(moves) == 11
        assert {"g6", "h7"}.isdisjoint(destinations(moves))

    def test_stops_on_enemy_capture(self):
        board = make_board(("e4", WHITE, Kind.BISHOP), ("g6", BLACK, Kind.PAWN))
        moves = bishop_moves(board, to_square("e4"))
        assert len(moves) == 12
        assert "h7" not in destinations(moves)
        assert find_move(moves, "g6").captured == Piece(BLACK, Kind.PAWN)


class TestQueen:
    def test_empty_board(self):
        board = make_board(("e4", WHITE, Kind.QUEEN))
        assert len(queen_moves(board, to_square("e4"))) == 27

    def test_start_position_has_no_moves(self):
        board = Board.starting_position()
        assert queen_moves(board, to_square("d1")) == []


class TestPawn:
    @pytest.mark.parametrize("col", range(8))
    def test_start_position_white_pawns(self, col):
        board = Board.starting_position()
        assert len(pawn_moves(board, (6, col))) == 2

    @pytest.mark.parametrize("col", range(8))
    def test_start_position_black_pawns(self, col):
        board = Board.starting_position()
        assert len(pawn_moves(board, (1, col))) == 2

    def test_white_double_push(self):
        board = make_board(("e2", WHITE, Kind.PAWN))
        assert destinations(pawn_moves(board, to_square("e2"))) == {"e3", "e4"}

    def test_black_double_push(self):
        board = make_board(("e7", BLACK, Kind.PAWN))
        assert destinations(pawn_moves(board, to_square("e7"))) == {"e6", "e5"}

    def test_white_single_push_off_start_row(self):
        board = make_board(("e3", WHITE, Kind.PAWN))
        assert destinations(pawn_moves(board, to_square("e3"))) == {"e4"}

    def test_black_moves_down_the_board(self):
        board = make_board(("e6", BLACK, Kind.PAWN))
        assert destinations(pawn_moves(board, to_square("e6"))) == {"e5"}

    def test_blocked_by_piece_in_front(self):
        board = make_board(("e2", WHITE, Kind.PAWN), ("e3", BLACK, Kind.KNIGHT))
        assert pawn_moves(board, to_square("e2")) == []

    def test_double_push_blocked_on_second_square(self):
        board = make_board(("e2", WHITE, Kind.PAWN), ("e4", BLACK, Kind.KNIGHT))
        assert destinations(pawn_moves(board, to_square("e2"))) == {"e3"}

    def test_cannot_capture_forward(self):
        board = make_board(("e4", WHITE, Kind.PAWN), ("e5", BLACK, Kind.PAWN))
        assert pawn_moves(board, to_square("e4")) == []

    def test_diagonal_captures(self):
        board = make_board(
            ("e4", WHITE, Kind.PAWN),
            ("d5", BLACK, Kind.PAWN),
            ("f5", BLACK, Kind.KNIGHT),
        )
        moves = pawn_moves(board, to_square("e4"))
        assert destinations(moves) == {"e5", "d5", "f5"}
        assert find_move(moves, "d5").captured == Piece(BLACK, Kind.PAWN)
        assert find_move(moves, "f5").captured == Piece(BLACK, Kind.KNIGHT)

    def test_no_diagonal_move_to_empty_square(self):
        board = make_board(("e4", WHITE, Kind.PAWN))
        assert destinations(pawn_moves(board, to_square("e4"))) == {"e5"}

    def test_cannot_capture_ally_diagonally(self):
        board = make_board(("e4", WHITE, Kind.PAWN), ("d5", WHITE, Kind.PAWN))
        assert destinations(pawn_moves(board, to_square("e4"))) == {"e5"}

    def test_black_diagonal_capture(self):
        board = make_board(("e5", BLACK, Kind.PAWN), ("d4", WHITE, Kind.PAWN))
        assert destinations(pawn_moves(board, to_square("e5"))) == {"e4", "d4"}

    def test_capture_on_left_edge(self):
        board = make_board(("a4", WHITE, Kind.PAWN), ("b5", BLACK, Kind.PAWN))
        assert destinations(pawn_moves(board, to_square("a4"))) == {"a5", "b5"}

    def test_capture_on_right_edge(self):
        board = make_board(("h4", WHITE, Kind.PAWN), ("g5", BLACK, Kind.PAWN))
        assert destinations(pawn_moves(board, to_square("h4"))) == {"h5", "g5"}

    def test_pawn_on_last_rank_has_no_moves(self):
        board = make_board(("e8", WHITE, Kind.PAWN), ("d1", BLACK, Kind.KNIGHT))
        assert pawn_moves(board, to_square("e8")) == []

    def test_non_pawn_or_empty_square_returns_nothing(self):
        board = make_board(("e4", WHITE, Kind.KNIGHT))
        assert pawn_moves(board, to_square("e4")) == []
        assert pawn_moves(board, to_square("a1")) == []


class TestGeneratePseudoLegalMoves:
    def test_start_position_white(self):
        board = Board.starting_position()
        assert len(generate_pseudo_legal_moves(board, WHITE)) == 20

    def test_start_position_black(self):
        board = Board.starting_position()
        assert len(generate_pseudo_legal_moves(board, BLACK)) == 20

    def test_only_returns_moves_of_requested_color(self):
        board = make_board(("e4", WHITE, Kind.KNIGHT), ("e5", BLACK, Kind.KING))
        white_moves = generate_pseudo_legal_moves(board, WHITE)
        assert {from_square(move.start) for move in white_moves} == {"e4"}

    def test_after_e4_e5_white_has_29_moves(self):
        board = Board.starting_position()
        board.make_move(Move(to_square("e2"), to_square("e4")))
        board.make_move(Move(to_square("e7"), to_square("e5")))
        assert len(generate_pseudo_legal_moves(board, WHITE)) == 29

    def test_moves_are_playable_and_undoable(self):
        board = Board.starting_position()
        before = [row[:] for row in board.grid]
        for move in generate_pseudo_legal_moves(board, WHITE):
            board.make_move(move)
            board.undo_move()
            assert board.grid == before
            assert board.turn == WHITE


class TestIsSquareAttacked:
    def test_knight(self):
        board = make_board(("f3", BLACK, Kind.KNIGHT))
        assert is_square_attacked(board, to_square("e1"), BLACK)

    def test_rook_on_open_file(self):
        board = make_board(("e8", BLACK, Kind.ROOK))
        assert is_square_attacked(board, to_square("e1"), BLACK)

    def test_rook_blocked_by_piece(self):
        board = make_board(("e8", BLACK, Kind.ROOK), ("e4", WHITE, Kind.PAWN))
        assert not is_square_attacked(board, to_square("e1"), BLACK)

    def test_bishop_diagonal(self):
        board = make_board(("a5", BLACK, Kind.BISHOP))
        assert is_square_attacked(board, to_square("e1"), BLACK)

    def test_queen(self):
        board = make_board(("e8", BLACK, Kind.QUEEN))
        assert is_square_attacked(board, to_square("e1"), BLACK)

    def test_king(self):
        board = make_board(("e3", BLACK, Kind.KING))
        assert is_square_attacked(board, to_square("f2"), BLACK)

    def test_black_pawn_attacks_diagonally_only(self):
        board = make_board(("d2", BLACK, Kind.PAWN))
        assert is_square_attacked(board, to_square("e1"), BLACK)
        assert is_square_attacked(board, to_square("c1"), BLACK)
        assert not is_square_attacked(board, to_square("d1"), BLACK)

    def test_white_pawn_attacks_diagonally_only(self):
        board = make_board(("e4", WHITE, Kind.PAWN))
        assert is_square_attacked(board, to_square("d5"), WHITE)
        assert is_square_attacked(board, to_square("f5"), WHITE)
        assert not is_square_attacked(board, to_square("e5"), WHITE)

    def test_attacker_color_matters(self):
        board = make_board(("f3", BLACK, Kind.KNIGHT))
        assert not is_square_attacked(board, to_square("e1"), WHITE)


class TestIsInCheck:
    def test_start_position(self):
        board = Board.starting_position()
        assert not is_in_check(board, WHITE)
        assert not is_in_check(board, BLACK)

    def test_check_by_rook(self):
        board = make_board(("e1", WHITE, Kind.KING), ("e8", BLACK, Kind.ROOK))
        assert is_in_check(board, WHITE)

    def test_no_check_when_blocked(self):
        board = make_board(
            ("e1", WHITE, Kind.KING),
            ("e8", BLACK, Kind.ROOK),
            ("e4", WHITE, Kind.PAWN),
        )
        assert not is_in_check(board, WHITE)

    def test_missing_king_raises(self):
        board = make_board(("e4", WHITE, Kind.PAWN))
        with pytest.raises(ValueError):
            is_in_check(board, WHITE)


class TestGenerateLegalMoves:
    def test_start_position(self):
        board = Board.starting_position()
        assert len(generate_legal_moves(board, WHITE)) == 20

    def test_king_cannot_move_into_attack(self):
        board = make_board(("e1", WHITE, Kind.KING), ("d8", BLACK, Kind.ROOK))
        moves = generate_legal_moves(board, WHITE)
        assert destinations(moves) == {"e2", "f1", "f2"}

    def test_king_cannot_capture_protected_piece(self):
        board = make_board(
            ("e4", WHITE, Kind.KING),
            ("e5", BLACK, Kind.PAWN),
            ("d6", BLACK, Kind.PAWN),
        )
        moves = generate_legal_moves(board, WHITE)
        assert destinations(moves) == {"d3", "e3", "f3", "d5", "f5"}

    def test_pinned_piece_stays_on_the_line(self):
        board = make_board(
            ("e1", WHITE, Kind.KING),
            ("e2", WHITE, Kind.ROOK),
            ("e8", BLACK, Kind.ROOK),
        )
        moves = [m for m in generate_legal_moves(board, WHITE) if from_square(m.start) == "e2"]
        assert destinations(moves) == {"e3", "e4", "e5", "e6", "e7", "e8"}

    def test_must_resolve_check(self):
        board = make_board(
            ("e1", WHITE, Kind.KING),
            ("c3", WHITE, Kind.BISHOP),
            ("e8", BLACK, Kind.ROOK),
        )
        moves = [m for m in generate_legal_moves(board, WHITE) if from_square(m.start) == "c3"]
        assert destinations(moves) == {"e5"}

    def test_no_legal_move_leaves_king_in_check(self):
        board = make_board(
            ("e1", WHITE, Kind.KING),
            ("c3", WHITE, Kind.BISHOP),
            ("e8", BLACK, Kind.ROOK),
        )
        for move in generate_legal_moves(board, WHITE):
            board.make_move(move)
            assert not is_in_check(board, WHITE)
            board.undo_move()

    def test_board_is_unchanged_after_generation(self):
        board = Board.starting_position()
        before = [row[:] for row in board.grid]
        generate_legal_moves(board, WHITE)
        assert board.grid == before
        assert board.turn == WHITE
        assert board.history == []


def play(board: Board, *moves: str) -> None:
    for notation in moves:
        start, end = to_square(notation[:2]), to_square(notation[2:4])
        board.make_move(Move(start, end, captured=board.piece_at(end)))


class TestGameEnd:
    def test_start_position_is_neither(self):
        board = Board.starting_position()
        assert not is_checkmate(board, WHITE)
        assert not is_stalemate(board, WHITE)

    def test_fools_mate(self):
        board = Board.starting_position()
        play(board, "f2f3", "e7e5", "g2g4", "d8h4")
        assert is_checkmate(board, WHITE)
        assert not is_stalemate(board, WHITE)

    def test_back_rank_mate(self):
        board = make_board(
            ("g8", BLACK, Kind.KING),
            ("f7", BLACK, Kind.PAWN),
            ("g7", BLACK, Kind.PAWN),
            ("h7", BLACK, Kind.PAWN),
            ("a8", WHITE, Kind.ROOK),
        )
        assert is_checkmate(board, BLACK)
        assert not is_stalemate(board, BLACK)

    def test_check_with_escape_is_not_checkmate(self):
        board = make_board(("e1", WHITE, Kind.KING), ("e8", BLACK, Kind.ROOK))
        assert is_in_check(board, WHITE)
        assert not is_checkmate(board, WHITE)
        assert not is_stalemate(board, WHITE)

    def test_stalemate(self):
        board = make_board(
            ("a8", BLACK, Kind.KING),
            ("c7", WHITE, Kind.KING),
            ("b6", WHITE, Kind.QUEEN),
        )
        assert is_stalemate(board, BLACK)
        assert not is_checkmate(board, BLACK)


class TestPawnPromotion:
    def test_white_push_generates_four_promotions(self):
        board = make_board(("e7", WHITE, Kind.PAWN))
        moves = pawn_moves(board, to_square("e7"))
        assert len(moves) == 4
        assert destinations(moves) == {"e8"}
        assert {move.promotion for move in moves} == EXPECTED_PROMOTIONS

    def test_black_push_generates_four_promotions(self):
        board = make_board(("e2", BLACK, Kind.PAWN))
        moves = pawn_moves(board, to_square("e2"))
        assert len(moves) == 4
        assert destinations(moves) == {"e1"}
        assert {move.promotion for move in moves} == EXPECTED_PROMOTIONS

    def test_capture_generates_four_promotions(self):
        board = make_board(("e7", WHITE, Kind.PAWN), ("d8", BLACK, Kind.ROOK))
        moves = pawn_moves(board, to_square("e7"))
        captures = [move for move in moves if move.captured is not None]
        assert len(moves) == 8
        assert len(captures) == 4
        assert all(from_square(move.end) == "d8" for move in captures)
        assert {move.promotion for move in captures} == EXPECTED_PROMOTIONS

    def test_blocked_pawn_has_no_promotion(self):
        board = make_board(("e7", WHITE, Kind.PAWN), ("e8", BLACK, Kind.KNIGHT))
        assert pawn_moves(board, to_square("e7")) == []

    def test_regular_pawn_move_has_no_promotion(self):
        board = make_board(("e6", WHITE, Kind.PAWN))
        moves = pawn_moves(board, to_square("e6"))
        assert all(move.promotion is None for move in moves)


class TestEnPassant:
    def test_white_captures_on_the_left(self):
        board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        moves = pawn_moves(board, to_square("e5"))
        assert destinations(moves) == {"e6", "d6"}
        capture = find_move(moves, "d6")
        assert capture.flag == "en_passant"
        assert capture.captured == Piece(BLACK, Kind.PAWN)

    def test_white_captures_on_the_right(self):
        board = Board.from_fen("4k3/8/8/4Pp2/8/8/8/4K3 w - f6 0 1")
        assert destinations(pawn_moves(board, to_square("e5"))) == {"e6", "f6"}

    def test_black_captures_en_passant(self):
        board = Board.from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
        moves = pawn_moves(board, to_square("d4"))
        assert destinations(moves) == {"d3", "e3"}
        capture = find_move(moves, "e3")
        assert capture.flag == "en_passant"
        assert capture.captured == Piece(WHITE, Kind.PAWN)

    def test_no_en_passant_without_target_square(self):
        board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1")
        assert destinations(pawn_moves(board, to_square("e5"))) == {"e6"}

    def test_pawn_cannot_use_en_passant_square_of_its_own_side(self):
        board = Board.from_fen("4k3/8/8/8/4P3/8/3P4/4K3 b - e3 0 1")
        assert destinations(pawn_moves(board, to_square("d2"))) == {"d3", "d4"}

    def test_illegal_when_it_exposes_king_on_the_rank(self):
        board = Board.from_fen("8/8/8/8/k2Pp2Q/8/8/4K3 b - d3 0 1")
        moves = [m for m in generate_legal_moves(board, BLACK) if from_square(m.start) == "e4"]
        assert destinations(moves) == {"e3"}


def castling_flags(fen: str, king: str) -> set[str]:
    board = Board.from_fen(fen)
    return {m.flag for m in king_moves(board, to_square(king)) if m.flag is not None}


class TestCastling:
    def test_white_both_sides(self):
        board = Board.from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
        moves = [m for m in king_moves(board, to_square("e1")) if m.flag is not None]
        assert {m.flag for m in moves} == {"castle_k", "castle_q"}
        assert destinations(moves) == {"g1", "c1"}

    def test_black_both_sides(self):
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1")
        moves = [m for m in king_moves(board, to_square("e8")) if m.flag is not None]
        assert {m.flag for m in moves} == {"castle_k", "castle_q"}
        assert destinations(moves) == {"g8", "c8"}

    def test_no_rights(self):
        assert castling_flags("4k3/8/8/8/8/8/8/R3K2R w - - 0 1", "e1") == set()

    def test_only_kingside_right(self):
        assert castling_flags("4k3/8/8/8/8/8/8/R3K2R w K - 0 1", "e1") == {"castle_k"}

    def test_blocked_kingside(self):
        assert castling_flags("4k3/8/8/8/8/8/8/R3K1NR w KQ - 0 1", "e1") == {"castle_q"}

    def test_blocked_queenside(self):
        assert castling_flags("4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1", "e1") == {"castle_k"}

    def test_cannot_castle_out_of_check(self):
        assert castling_flags("4r1k1/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1") == set()

    def test_cannot_castle_through_attacked_square(self):
        assert castling_flags("5rk1/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1") == {"castle_q"}

    def test_cannot_castle_onto_attacked_square(self):
        assert castling_flags("6r1/7k/8/8/8/8/8/R3K2R w KQ - 0 1", "e1") == {"castle_q"}

    def test_queenside_allowed_when_b1_is_attacked(self):
        assert castling_flags("1r2k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1") == {"castle_k", "castle_q"}

    def test_requires_rook_on_corner(self):
        assert castling_flags("4k3/8/8/8/8/8/8/4K2R w KQ - 0 1", "e1") == {"castle_k"}

    def test_king_must_be_on_home_square(self):
        assert castling_flags("4k3/8/8/8/8/8/8/R4K1R w KQ - 0 1", "f1") == set()