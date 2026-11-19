import numpy as np

PIECE_ORDER = "PNBRQK"
PLANES = 2 * len(PIECE_ORDER)
BOARD_CELLS = PLANES * 64
PACKED_BYTES = BOARD_CELLS // 8


def parse_fen(fen: str) -> tuple[list[list[str]], bool]:
    fields = fen.split()
    rows: list[list[str]] = []
    for rank in fields[0].split("/"):
        row: list[str] = []
        for char in rank:
            if char.isdigit():
                row.extend(["."] * int(char))
            else:
                row.append(char)
        rows.append(row)
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
        raise ValueError(f"Invalid piece placement in FEN: {fen}")
    return rows, fields[1] == "w"


def _square(text: str) -> tuple[int, int]:
    file = ord(text[0]) - ord("a")
    rank = int(text[1])
    if not (0 <= file < 8 and 1 <= rank <= 8):
        raise ValueError(f"Invalid square: {text}")
    return 8 - rank, file


def apply_uci(rows: list[list[str]], uci: str) -> None:
    from_row, from_file = _square(uci[0:2])
    to_row, to_file = _square(uci[2:4])
    piece = rows[from_row][from_file]
    if piece == ".":
        raise ValueError(f"No piece on {uci[0:2]}")
    if piece in "Pp" and from_file != to_file and rows[to_row][to_file] == ".":
        rows[from_row][to_file] = "."
    if piece in "Kk" and abs(to_file - from_file) == 2:
        if to_file > from_file:
            rows[to_row][to_file - 1] = rows[to_row][7]
            rows[to_row][7] = "."
        else:
            rows[to_row][to_file + 1] = rows[to_row][0]
            rows[to_row][0] = "."
    if len(uci) == 5:
        piece = uci[4].upper() if piece.isupper() else uci[4].lower()
    rows[from_row][from_file] = "."
    rows[to_row][to_file] = piece


def encode_position(rows: list[list[str]], white_to_move: bool) -> np.ndarray:
    planes = np.zeros((PLANES, 8, 8), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            piece = rows[row][col]
            if piece == ".":
                continue
            own = piece.isupper() == white_to_move
            target_row, target_col = (row, col) if white_to_move else (7 - row, col)
            plane = PIECE_ORDER.index(piece.upper()) + (0 if own else len(PIECE_ORDER))
            planes[plane, target_row, target_col] = 1
    return planes


def pack_positions(planes: np.ndarray) -> np.ndarray:
    return np.packbits(planes.reshape(len(planes), BOARD_CELLS), axis=1)


def unpack_positions(packed: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(packed, axis=1)[:, :BOARD_CELLS]
    return bits.reshape(len(packed), PLANES, 8, 8)