from chess.game import Game, GameStatus
from chess.move import Move, Square
from chess.pieces import Color, Kind
from chess.rules import is_in_check


class BoardController:
    def __init__(self, game: Game, human_color: Color | None = None) -> None:
        self.game = game
        self.human_color = human_color
        self.selected: Square | None = None
        self.moves_from_selected: list[Move] = []
        self.pending_promotion: list[Move] = []
        self.status: GameStatus = GameStatus.ONGOING
        self.in_check: bool = False
        self._refresh()

    @property
    def destinations(self) -> set[Square]:
        return {move.end for move in self.moves_from_selected}

    @property
    def is_human_turn(self) -> bool:
        return self.human_color is None or self.game.board.turn == self.human_color
    
    @property
    def is_bot_turn(self) -> bool:
        return (
            self.human_color is not None
            and self.status == GameStatus.ONGOING
            and self.game.board.turn != self.human_color
        )

    def click(self, square: Square) -> None:
        if self.status != GameStatus.ONGOING or self.pending_promotion or not self.is_human_turn:
            return
        if self.selected is not None and square in self.destinations:
            candidates = [move for move in self.moves_from_selected if move.end == square]
            if len(candidates) == 1:
                self._play(candidates[0])
            else:
                self.pending_promotion = candidates
            return
        piece = self.game.board.piece_at(square)
        if piece is not None and piece.color == self.game.board.turn:
            if square == self.selected:
                self._deselect()
            else:
                self._select(square)
        else:
            self._deselect()

    def choose_promotion(self, kind: Kind) -> None:
        for move in self.pending_promotion:
            if move.promotion == kind:
                self._play(move)
                return

    def _select(self, square: Square) -> None:
        self.selected = square
        self.moves_from_selected = [move for move in self.game.legal_moves() if move.start == square]

    def _deselect(self) -> None:
        self.selected = None
        self.moves_from_selected = []

    def cancel_promotion(self) -> None:
        self.pending_promotion = []
        self._deselect()

    def _refresh(self) -> None:
        self.status = self.game.status()
        self.in_check = is_in_check(self.game.board, self.game.board.turn)

    def _play(self, move: Move) -> None:
        self.game.play(move)
        self._deselect()
        self.pending_promotion = []
        self._refresh()
    
    def play_move(self, move: Move) -> None:
        self._play(move)