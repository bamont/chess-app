import random

from bot.evaluation import evaluate
from chess.board import Board
from chess.move import Move
from chess.rules import generate_legal_moves


class GreedyBot:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, board: Board) -> Move:
        moves = generate_legal_moves(board, board.turn)
        if not moves:
            raise ValueError("No legal moves available")
        best_score = None
        best_moves: list[Move] = []
        for move in moves:
            board.make_move(move)
            score = -evaluate(board)
            board.undo_move()
            if best_score is None or score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        return self.rng.choice(best_moves)