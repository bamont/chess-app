import random
from collections.abc import Iterable
from functools import lru_cache

from chess.board import Board
from chess.move import Move
from chess.notation import parse_uci
from chess.rules import generate_legal_moves

BOOK_LINES = (
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6 e1g1 e8g8",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4 d6d5 f1d3 b8c6 e1g1 f8e7",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 d4c6 b7c6 e4e5 d8e7 d1e2 f6d5",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6 f1e2 e7e6 e1g1 f8e7",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 f8e7 e4e5 f6d7 g5e7 d8e7 f2f4 e8g8",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6 g1f3 b8d7",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 h7h6 g5h4 b7b6",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5 e2e3 e7e6 f1c4 f8b4",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5 g1f3 c7c5 e1g1 b8c6",
    "d2d4 d7d5 c1f4 g8f6 e2e3 e7e6 g1f3 c7c5 c2c3 b8c6 b1d2 f8d6 f4g3 e8g8",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6 e1g1 f8e7",
    "c2c4 g8f6 b1c3 e7e6 g1f3 d7d5 d2d4 f8e7 c1f4 e8g8 e2e3 c7c5 d4c5 e7c5",
    "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6 e1g1 f8e7 d2d3 e8g8 b1d2 c7c5 e2e4 b8c6",
    "g1f3 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 e8g8 d2d4 d7d6 f1e2 e7e5 e1g1",
)


def book_key(board: Board) -> tuple:
    return board.position_key()[:3]


class OpeningBook:
    def __init__(self, lines: Iterable[str] = BOOK_LINES) -> None:
        self._replies: dict[tuple, list[str]] = {}
        for line in lines:
            self._add_line(line)

    def __len__(self) -> int:
        return len(self._replies)

    def choose(self, board: Board, rng: random.Random) -> Move | None:
        replies = self._replies.get(book_key(board))
        if not replies:
            return None
        legal_moves = generate_legal_moves(board, board.turn)
        return parse_uci(rng.choice(replies), legal_moves)

    def _add_line(self, line: str) -> None:
        board = Board.starting_position()
        for text in line.split():
            move = parse_uci(text, generate_legal_moves(board, board.turn))
            replies = self._replies.setdefault(book_key(board), [])
            if text not in replies:
                replies.append(text)
            board.make_move(move)


@lru_cache(maxsize=1)
def default_book() -> OpeningBook:
    return OpeningBook()