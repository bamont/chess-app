import random

import pytest

from bot.minimax_bot import MinimaxBot
from bot.opening_book import BOOK_LINES, OpeningBook, default_book
from chess.board import Board
from chess.notation import move_to_uci
from chess.rules import generate_legal_moves

FIRST_MOVES = ("e2e4", "d2d4", "c2c4", "g1f3")


def play(board: Board, uci: str) -> None:
    move = next(m for m in generate_legal_moves(board, board.turn) if move_to_uci(m) == uci)
    board.make_move(move)


def test_every_line_is_legal():
    assert len(OpeningBook(BOOK_LINES)) > 0


def test_invalid_line_raises():
    with pytest.raises(ValueError):
        OpeningBook(["e2e5"])


def test_start_position_offers_the_main_first_moves():
    book = OpeningBook()
    rng = random.Random(0)
    first_moves = {move_to_uci(book.choose(Board.starting_position(), rng)) for _ in range(200)}
    assert first_moves == set(FIRST_MOVES)


@pytest.mark.parametrize("first_move", FIRST_MOVES)
def test_black_has_an_answer_to_every_main_first_move(first_move):
    board = Board.starting_position()
    play(board, first_move)
    move = OpeningBook().choose(board, random.Random(0))
    assert move is not None
    assert move in generate_legal_moves(board, board.turn)


def test_out_of_book_position_returns_none():
    board = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert OpeningBook().choose(board, random.Random(0)) is None


def test_transpositions_are_recognized():
    book = OpeningBook(["e2e4 e7e5 g1f3 b8c6 f1b5"])
    board = Board.starting_position()
    for uci in ("g1f3", "b8c6", "e2e4", "e7e5"):
        play(board, uci)
    assert move_to_uci(book.choose(board, random.Random(0))) == "f1b5"


def test_default_book_is_shared():
    assert default_book() is default_book()


def test_bot_plays_a_book_move_from_the_start():
    bot = MinimaxBot(depth=2, seed=0, book=OpeningBook())
    move = bot.choose_move(Board.starting_position())
    assert move_to_uci(move) in FIRST_MOVES
    assert bot.last_score is None


def test_bot_searches_when_out_of_book():
    board = Board.from_fen("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    bot = MinimaxBot(depth=2, seed=0, book=OpeningBook())
    assert move_to_uci(bot.choose_move(board)) == "a1a8"