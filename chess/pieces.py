from enum import Enum
from typing import NamedTuple


class Color(str, Enum):
    WHITE = "w"
    BLACK = "b"

    def opponent(self) -> "Color":
        return Color.BLACK if self is Color.WHITE else Color.WHITE


class Kind(str, Enum):
    PAWN = "P"
    KNIGHT = "N"
    BISHOP = "B"
    ROOK = "R"
    QUEEN = "Q"
    KING = "K"


class Piece(NamedTuple):
    color: Color
    kind: Kind