from enum import Enum

from chess.board import Board
from chess.move import Move
from chess.pieces import Color
from chess.rules import generate_legal_moves, has_insufficient_material, is_in_check


class GameStatus(Enum):
    ONGOING = "ongoing"
    WHITE_WINS = "white_wins"
    BLACK_WINS = "black_wins"
    STALEMATE = "stalemate"
    DRAW_FIFTY_MOVES = "draw_fifty_moves"
    DRAW_REPETITION = "draw_repetition"
    DRAW_INSUFFICIENT_MATERIAL = "draw_insufficient_material"


class Game:
    def __init__(self, board: Board | None = None) -> None:
        self.board = board if board is not None else Board.starting_position()
        self.position_history: list[tuple] = [self.board.position_key()]

    def legal_moves(self) -> list[Move]:
        return generate_legal_moves(self.board, self.board.turn)

    def play(self, move: Move) -> None:
        if move not in self.legal_moves():
            raise ValueError(f"Illegal move: {move}")
        self.board.make_move(move)
        self.position_history.append(self.board.position_key())

    def status(self) -> GameStatus:
        if not self.legal_moves():
            if is_in_check(self.board, self.board.turn):
                return GameStatus.WHITE_WINS if self.board.turn == Color.BLACK else GameStatus.BLACK_WINS
            return GameStatus.STALEMATE
        if has_insufficient_material(self.board):
            return GameStatus.DRAW_INSUFFICIENT_MATERIAL
        if self.board.halfmove_clock >= 100:
            return GameStatus.DRAW_FIFTY_MOVES
        if self.position_history.count(self.board.position_key()) >= 3:
            return GameStatus.DRAW_REPETITION
        return GameStatus.ONGOING