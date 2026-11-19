import sys
from pathlib import Path

import pytest

from bot.random_bot import RandomBot
from chess.board import Board
from chess.game import GameStatus
from chess.notation import move_to_uci
from chess.pieces import Color
from chess.rules import generate_legal_moves
from tools.match import play_game
from tools.uci_bot import UciBot
from tools.uci_engine import UciEngine, parse_option

FAKE_ENGINE = [sys.executable, str(Path(__file__).parent / "fake_uci_engine.py")]


def legal_ucis(board: Board) -> set[str]:
    return {move_to_uci(move) for move in generate_legal_moves(board, board.turn)}


@pytest.fixture
def engine():
    with UciEngine(FAKE_ENGINE) as running:
        yield running


def test_parse_spin_option():
    line = "option name UCI_Elo type spin default 1320 min 1320 max 3190"
    assert parse_option(line) == ("UCI_Elo", 1320, 3190)


def test_parse_option_with_spaces_in_its_name():
    line = "option name Skill Level type spin default 20 min 0 max 20"
    assert parse_option(line) == ("Skill Level", 0, 20)


def test_parse_check_option_has_no_range():
    assert parse_option("option name UCI_LimitStrength type check default false") == (
        "UCI_LimitStrength",
        None,
        None,
    )


def test_parse_ignores_other_lines():
    assert parse_option("id name Stockfish 17") is None


def test_handshake_reads_option_ranges(engine):
    assert engine.has_option("UCI_LimitStrength")
    assert engine.option_range("UCI_Elo") == (1320, 3190)
    assert engine.option_range("UCI_LimitStrength") is None
    assert not engine.has_option("Nope")


def test_best_move_from_the_start_position(engine):
    assert engine.best_move([], 10) in legal_ucis(Board.starting_position())


def test_best_move_after_some_moves(engine):
    board = Board.starting_position()
    board.make_move(next(m for m in generate_legal_moves(board, board.turn) if move_to_uci(m) == "e2e4"))
    assert engine.best_move(["e2e4"], 10) in legal_ucis(board)


def test_options_can_be_set(engine):
    engine.set_option("UCI_LimitStrength", "true")
    engine.set_option("UCI_Elo", "1500")
    engine.new_game()


def test_uci_bot_plays_a_legal_game():
    with UciEngine(FAKE_ENGINE) as engine:
        bot = UciBot(engine, movetime_ms=10)
        status, plies = play_game(bot, RandomBot(seed=1), max_plies=24, opening_plies=2, opening_seed=5)
    assert plies <= 24
    assert isinstance(status, GameStatus)


def test_uci_bot_can_play_black():
    with UciEngine(FAKE_ENGINE) as engine:
        bot = UciBot(engine, movetime_ms=10)
        status, plies = play_game(RandomBot(seed=2), bot, max_plies=24, opening_plies=2, opening_seed=6)
    assert plies <= 24
    assert isinstance(status, GameStatus)


def test_uci_bot_rejects_a_game_not_started_from_the_standard_position():
    with UciEngine(FAKE_ENGINE) as engine:
        bot = UciBot(engine, movetime_ms=10)
        with pytest.raises(ValueError):
            bot.choose_move(Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))


def test_the_turn_after_the_engine_move_belongs_to_the_other_side():
    with UciEngine(FAKE_ENGINE) as engine:
        bot = UciBot(engine, movetime_ms=10)
        board = Board.starting_position()
        board.make_move(bot.choose_move(board))
        assert board.turn == Color.BLACK