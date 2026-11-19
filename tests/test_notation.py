import pytest

from chess.board import Board
from chess.game import Game
from chess.notation import move_to_uci, parse_uci
from chess.pieces import Kind
from chess.utils import to_square


def legal_moves(fen: str):
    return Game(Board.from_fen(fen)).legal_moves()


class TestMoveToUci:
    def test_start_position(self):
        texts = {move_to_uci(move) for move in Game().legal_moves()}
        assert len(texts) == 20
        assert {"e2e4", "g1f3"} <= texts

    def test_promotion_has_letter(self):
        moves = legal_moves("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        texts = {move_to_uci(m) for m in moves if m.start == to_square("a7")}
        assert texts == {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}


class TestParseUci:
    def test_valid_move(self):
        move = parse_uci("e2e4", Game().legal_moves())
        assert move.start == to_square("e2")
        assert move.end == to_square("e4")

    def test_ignores_case_and_whitespace(self):
        assert parse_uci("  E2E4 ", Game().legal_moves()).end == to_square("e4")

    @pytest.mark.parametrize("text", ["", "e2", "e2e4e5", "z2e4", "e9e4", "hello", "e2-e4"])
    def test_malformed_input_raises(self, text):
        with pytest.raises(ValueError):
            parse_uci(text, Game().legal_moves())

    def test_illegal_move_raises(self):
        with pytest.raises(ValueError):
            parse_uci("e2e5", Game().legal_moves())

    def test_promotion_with_letter(self):
        moves = legal_moves("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        assert parse_uci("a7a8n", moves).promotion == Kind.KNIGHT

    def test_promotion_without_letter_raises(self):
        moves = legal_moves("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        with pytest.raises(ValueError):
            parse_uci("a7a8", moves)

    def test_promotion_letter_on_normal_move_raises(self):
        with pytest.raises(ValueError):
            parse_uci("e2e4q", Game().legal_moves())

    def test_castling(self):
        moves = legal_moves("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
        assert parse_uci("e1g1", moves).flag == "castle_k"
        assert parse_uci("e1c1", moves).flag == "castle_q"

    def test_en_passant(self):
        moves = legal_moves("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        assert parse_uci("e5d6", moves).flag == "en_passant"