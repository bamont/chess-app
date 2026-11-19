from chess.move import Square


def to_square(notation: str) -> Square:
    col = ord(notation[0]) - ord("a")
    row = 8 - int(notation[1])
    return row, col

def from_square(square: Square) -> str:
    row, col = square
    return f"{chr(col + ord('a'))}{8 - row}"