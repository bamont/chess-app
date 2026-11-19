from chess.board import Board
from chess.move import Move
from chess.notation import move_to_uci, parse_uci
from chess.rules import generate_legal_moves
from tools.uci_engine import UciEngine


class UciBot:
    def __init__(self, engine: UciEngine, movetime_ms: int) -> None:
        self.engine = engine
        self.movetime_ms = movetime_ms

    def choose_move(self, board: Board) -> Move:
        if not board.history and board.position_key() != Board.starting_position().position_key():
            raise ValueError("UciBot needs a game that started from the standard position")
        moves = [move_to_uci(entry.move) for entry in board.history]
        text = self.engine.best_move(moves, self.movetime_ms)
        return parse_uci(text, generate_legal_moves(board, board.turn))