import pytest

from bot.random_bot import RandomBot
from chess.board import Board
from chess.game import Game, GameStatus
from chess.pieces import Color
from chess.rules import find_king, generate_legal_moves, is_in_check


class TestChooseMove:
    def test_returns_a_legal_move(self):
        board = Board.starting_position()
        move = RandomBot(seed=0).choose_move(board)
        assert move in generate_legal_moves(board, board.turn)

    def test_same_seed_gives_same_choices(self):
        board = Board.starting_position()
        first, second = RandomBot(seed=7), RandomBot(seed=7)
        assert [first.choose_move(board) for _ in range(5)] == [second.choose_move(board) for _ in range(5)]

    def test_does_not_modify_the_board(self):
        board = Board.starting_position()
        before = board.position_key()
        RandomBot(seed=0).choose_move(board)
        assert board.position_key() == before
        assert board.history == []

    @pytest.mark.parametrize(
        "fen",
        [
            "R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1",
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        ],
    )
    def test_raises_when_no_legal_move(self, fen):
        with pytest.raises(ValueError):
            RandomBot(seed=0).choose_move(Board.from_fen(fen))


def play_random_game(seed: int, max_plies: int) -> Game:
    game = Game()
    bot = RandomBot(seed=seed)
    for _ in range(max_plies):
        if game.status() != GameStatus.ONGOING:
            break
        mover = game.board.turn
        game.play(bot.choose_move(game.board))
        assert not is_in_check(game.board, mover)
        for color in (Color.WHITE, Color.BLACK):
            find_king(game.board, color)
    return game


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_playout_then_undo_everything(seed):
    game = play_random_game(seed, max_plies=150)
    board = game.board
    reference = Board.starting_position()
    while board.history:
        board.undo_move()
    assert board.position_key() == reference.position_key()
    assert board.halfmove_clock == reference.halfmove_clock