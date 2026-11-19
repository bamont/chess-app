from typing import Protocol

from chess.board import Board
from chess.move import Move


class Bot(Protocol):
    def choose_move(self, board: Board) -> Move:
        ...