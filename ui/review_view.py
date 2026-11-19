import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from review.judge import KEY_MOMENT_LOSS, ReviewedMove, review_moves
from review.model import Candidate, PlyAnalysis
from review.winprob import format_score

SQUARE_PATTERN = re.compile(r"^[a-h][1-8]$")
ROW_HEIGHT = 22
SHOWN_ALTERNATIVES = 3
CATEGORY_TEXT = {
    "best": "best move",
    "excellent": "excellent",
    "good": "good",
    "inaccuracy": "inaccuracy ?!",
    "mistake": "mistake ?",
    "blunder": "blunder ??",
}
WEAK_SYMBOLS = {"inaccuracy": "?!", "mistake": "?", "blunder": "??"}

Square = tuple[int, int]
Squares = tuple[Square, Square]


@dataclass(frozen=True)
class ReviewRow:
    text: str
    best: str
    category: str


def _square(text: str) -> Square:
    return 8 - int(text[1]), ord(text[0]) - ord("a")


def uci_squares(uci: str) -> Squares | None:
    start, end = uci[0:2], uci[2:4]
    if not (SQUARE_PATTERN.match(start) and SQUARE_PATTERN.match(end)):
        return None
    return _square(start), _square(end)


def visible_start(cursor: int, total: int, rows: int) -> int:
    return max(0, min(cursor - rows // 2, max(0, total - rows)))


def row_at(y: int, top: int, start: int, rows: int, total: int) -> int | None:
    if y < top:
        return None
    index = start + (y - top) // ROW_HEIGHT
    if index >= start + rows or index >= total:
        return None
    return index


def arrow_shape(
    start: tuple[float, float], end: tuple[float, float], unit: float
) -> tuple[tuple[float, float], tuple[float, float], list[tuple[float, float]]] | None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    ux, uy = dx / length, dy / length
    head = unit * 0.45
    half = head * 0.6
    base = (end[0] - ux * head, end[1] - uy * head)
    side = (-uy * half, ux * half)
    points = [end, (base[0] + side[0], base[1] + side[1]), (base[0] - side[0], base[1] - side[1])]
    return start, base, points


def _board_from_fen(fen: str) -> Any:
    from chess.board import Board

    return Board.from_fen(fen)


class ReviewSession:
    def __init__(self, plies: Sequence[PlyAnalysis], make_board: Callable[[str], Any] | None = None) -> None:
        if not plies:
            raise ValueError("There is nothing to review")
        self.moves: list[ReviewedMove] = review_moves(list(plies))
        self.candidates: list[tuple[Candidate, ...]] = [tuple(ply.candidates) for ply in plies]
        self.cursor = 0
        self._make_board = make_board or _board_from_fen
        self._boards: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.moves)

    @property
    def current(self) -> ReviewedMove:
        return self.moves[self.cursor]

    @property
    def board(self) -> Any:
        if self.cursor not in self._boards:
            self._boards[self.cursor] = self._make_board(self.current.fen)
        return self._boards[self.cursor]

    def go_to(self, index: int) -> None:
        self.cursor = max(0, min(index, len(self.moves) - 1))

    def next(self) -> None:
        self.go_to(self.cursor + 1)

    def previous(self) -> None:
        self.go_to(self.cursor - 1)

    def first(self) -> None:
        self.go_to(0)

    def last(self) -> None:
        self.go_to(len(self.moves) - 1)

    def next_mistake(self) -> None:
        for index in range(self.cursor + 1, len(self.moves)):
            if self.moves[index].loss >= KEY_MOMENT_LOSS:
                self.cursor = index
                return

    def previous_mistake(self) -> None:
        for index in range(self.cursor - 1, -1, -1):
            if self.moves[index].loss >= KEY_MOMENT_LOSS:
                self.cursor = index
                return

    def evaluations(self) -> list[int]:
        return [move.white_cp_before for move in self.moves]

    @property
    def played_is_best(self) -> bool:
        return self.current.uci == self.current.best_uci

    @property
    def played_squares(self) -> Squares | None:
        return uci_squares(self.current.uci)

    @property
    def best_squares(self) -> list[Squares]:
        played = self.current.uci[:4]
        shapes = []
        choices = self.candidates[self.cursor]
        for candidate in choices[:SHOWN_ALTERNATIVES]:
            squares = None if candidate.uci[:4] == played else uci_squares(candidate.uci)
            if squares is not None:
                shapes.append(squares)
        if not choices and self.current.best_uci[:4] != played:
            squares = uci_squares(self.current.best_uci)
            if squares is not None:
                shapes.append(squares)
        return shapes

    def row(self, index: int) -> ReviewRow:
        move = self.moves[index]
        symbol = WEAK_SYMBOLS.get(move.category, "")
        text = f"{move.label} {symbol}".rstrip()
        best = "" if move.category in ("best", "excellent", "good") else f"best {move.best_san}"
        return ReviewRow(text, best, move.category)

    def headline(self) -> str:
        move = self.current
        text = f"{move.label}  {CATEGORY_TEXT[move.category]}"
        if move.loss >= 1:
            text += f" (-{move.loss:.0f} pts)"
        return text

    def detail_lines(self) -> list[str]:
        move = self.current
        lines = [f"Evaluation {format_score(move.white_cp_before)} -> {format_score(move.white_cp_after)}"]
        choices = self.candidates[self.cursor]
        if not choices:
            lines.append(f"Best: {move.best_san}")
            if len(move.best_line) > 1:
                lines.append(" ".join(move.best_line))
            return lines
        lines.append("Engine's choices:" if len(choices) > 1 else "Engine's choice:")
        for rank, candidate in enumerate(choices, start=1):
            marker = "  (played)" if candidate.uci[:4] == move.uci[:4] else ""
            lines.append(f"{rank}. {candidate.san}  {format_score(candidate.white_cp)}{marker}")
            if len(candidate.line) > 1:
                lines.append("    " + " ".join(candidate.line[1:]))
        if not any(candidate.uci[:4] == move.uci[:4] for candidate in choices):
            lines.append(f"Played: {move.san}  {format_score(move.white_cp_after)}")
        return lines