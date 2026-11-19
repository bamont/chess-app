from typing import NamedTuple

from chess.pieces import Kind, Piece

Square = tuple[int, int]


class Move(NamedTuple):
    start: Square
    end: Square
    captured: Piece | None = None
    promotion: Kind | None = None
    flag: str | None = None