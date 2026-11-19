import random

from chess.board import Board
from chess.move import Move
from chess.rules import generate_legal_moves


class RandomBot:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, board: Board) -> Move:
        legal_moves = generate_legal_moves(board, board.turn)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return self.rng.choice(legal_moves)