import threading
import time

import pytest

from bot.random_bot import RandomBot
from chess.board import Board
from chess.rules import generate_legal_moves
from ui.bot_player import BotPlayer


def wait_for_move(player: BotPlayer, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        move = player.poll()
        if move is not None:
            return move
        time.sleep(0.01)
    raise TimeoutError("the bot did not answer in time")


class GatedBot:
    def __init__(self) -> None:
        self.gate = threading.Event()
        self.board_seen = None

    def choose_move(self, board):
        self.gate.wait(timeout=2)
        self.board_seen = board
        return generate_legal_moves(board, board.turn)[0]


class FailingBot:
    def choose_move(self, board):
        raise RuntimeError("boom")


def test_returns_a_legal_move():
    board = Board.starting_position()
    player = BotPlayer(RandomBot(seed=1))
    player.start(board)
    assert wait_for_move(player) in generate_legal_moves(board, board.turn)


def test_start_does_not_block():
    bot = GatedBot()
    player = BotPlayer(bot)
    player.start(Board.starting_position())
    assert player.thinking
    assert player.poll() is None
    bot.gate.set()
    wait_for_move(player)
    assert not player.thinking


def test_bot_works_on_a_copy():
    board = Board.starting_position()
    bot = GatedBot()
    player = BotPlayer(bot)
    player.start(board)
    bot.gate.set()
    wait_for_move(player)
    assert bot.board_seen is not board


def test_errors_are_reraised_by_poll():
    player = BotPlayer(FailingBot())
    player.start(Board.starting_position())
    with pytest.raises(RuntimeError):
        wait_for_move(player)


def test_start_is_ignored_while_thinking():
    bot = GatedBot()
    player = BotPlayer(bot)
    player.start(Board.starting_position())
    player.start(Board.starting_position())
    bot.gate.set()
    wait_for_move(player)
    time.sleep(0.05)
    assert player.poll() is None