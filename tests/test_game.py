import pytest

from chess.board import Board
from chess.game import Game, GameStatus
from chess.utils import to_square


def play(game: Game, *notations: str) -> None:
    for notation in notations:
        start, end = to_square(notation[:2]), to_square(notation[2:4])
        move = next(m for m in game.legal_moves() if m.start == start and m.end == end)
        game.play(move)


def game_from_fen(fen: str) -> Game:
    return Game(Board.from_fen(fen))


class TestPlay:
    def test_new_game(self):
        game = Game()
        assert game.status() == GameStatus.ONGOING
        assert len(game.legal_moves()) == 20

    def test_play_changes_turn(self):
        game = Game()
        play(game, "e2e4")
        assert len(game.legal_moves()) == 20
        assert game.board.turn.value == "b"

    def test_illegal_move_is_rejected(self):
        game = Game()
        legal = game.legal_moves()
        illegal = next(m for m in Game(Board.from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")).legal_moves())
        assert illegal not in legal
        with pytest.raises(ValueError):
            game.play(illegal)


class TestCheckmateAndStalemate:
    def test_fools_mate_black_wins(self):
        game = Game()
        play(game, "f2f3", "e7e5", "g2g4", "d8h4")
        assert game.status() == GameStatus.BLACK_WINS

    def test_scholars_mate_white_wins(self):
        game = Game()
        play(game, "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7")
        assert game.status() == GameStatus.WHITE_WINS

    def test_stalemate(self):
        game = game_from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        assert game.status() == GameStatus.STALEMATE


class TestFiftyMoveRule:
    def test_draw_at_100_halfmoves(self):
        game = game_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 80")
        assert game.status() == GameStatus.DRAW_FIFTY_MOVES

    def test_not_a_draw_at_99(self):
        game = game_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 99 80")
        assert game.status() == GameStatus.ONGOING

    def test_checkmate_takes_precedence(self):
        game = game_from_fen("R5k1/5ppp/8/8/8/8/8/4K3 b - - 100 60")
        assert game.status() == GameStatus.WHITE_WINS


class TestRepetition:
    def test_threefold_repetition(self):
        game = Game()
        cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
        play(game, *cycle)
        assert game.status() == GameStatus.ONGOING
        play(game, *cycle)
        assert game.status() == GameStatus.DRAW_REPETITION


class TestInsufficientMaterial:
    @pytest.mark.parametrize(
        "fen",
        [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/1N2K3 w - - 0 1",
            "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",
        ],
    )
    def test_draw(self, fen):
        assert game_from_fen(fen).status() == GameStatus.DRAW_INSUFFICIENT_MATERIAL

    @pytest.mark.parametrize(
        "fen",
        [
            "2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
            "4k1n1/8/8/8/8/8/8/4K1N1 w - - 0 1",
            "4k3/8/8/8/8/8/8/3NKN2 w - - 0 1",
            "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        ],
    )
    def test_not_a_draw(self, fen):
        assert game_from_fen(fen).status() == GameStatus.ONGOING