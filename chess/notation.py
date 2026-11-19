import re

from chess.move import Move
from chess.pieces import Kind
from chess.utils import from_square, to_square

UCI_PATTERN = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")
PROMOTION_BY_LETTER = {
    "q": Kind.QUEEN,
    "r": Kind.ROOK,
    "b": Kind.BISHOP,
    "n": Kind.KNIGHT,
}


def move_to_uci(move: Move) -> str:
    start_square = from_square(move.start)
    end_square = from_square(move.end)
    promotion = ""
    if move.promotion is not None:
        promotion = move.promotion.value.lower()
    return f"{start_square}{end_square}{promotion}"


def parse_uci(text: str, legal_moves: list[Move]) -> Move:
    text = text.strip().lower()
    if not UCI_PATTERN.fullmatch(text):
        raise ValueError(f"Invalid move format: '{text}' (expected e.g. e2e4 or e7e8q)")
    start, end = to_square(text[:2]), to_square(text[2:4])
    promotion = PROMOTION_BY_LETTER[text[4]] if len(text) == 5 else None
    candidates = [move for move in legal_moves if move.start == start and move.end == end]
    if not candidates:
        raise ValueError(f"Illegal move: {text}")
    if promotion is None and candidates[0].promotion is not None:
        raise ValueError("Promotion piece required, e.g. e7e8q")
    for move in candidates:
        if move.promotion == promotion:
            return move
    raise ValueError(f"Illegal move: {text}")