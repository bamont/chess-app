import random
from typing import NamedTuple

from chess.move import Move, Square
from chess.pieces import Color, Kind, Piece
from chess.utils import from_square


class HistoryEntry(NamedTuple):
    move: Move
    moved_piece: Piece
    castling_rights: dict[str, bool]
    en_passant: Square | None
    halfmove_clock: int
    zobrist: int


CORNER_RIGHTS = {(7, 0): "wQ", (7, 7): "wK", (0, 0): "bQ", (0, 7): "bK"}
BACK_RANK = (
    Kind.ROOK,
    Kind.KNIGHT,
    Kind.BISHOP,
    Kind.QUEEN,
    Kind.KING,
    Kind.BISHOP,
    Kind.KNIGHT,
    Kind.ROOK,
)

_random = random.Random(20241023)
ZOBRIST_PIECES = {
    color: {kind: tuple(_random.getrandbits(64) for _ in range(64)) for kind in Kind} for color in Color
}
ZOBRIST_SIDE = _random.getrandbits(64)
ZOBRIST_CASTLING = tuple(_random.getrandbits(64) for _ in range(16))
ZOBRIST_EN_PASSANT = tuple(_random.getrandbits(64) for _ in range(8))


def _castling_mask(rights: dict[str, bool]) -> int:
    return rights["wK"] | (rights["wQ"] << 1) | (rights["bK"] << 2) | (rights["bQ"] << 3)


class Board:
    def __init__(self) -> None:
        self.grid: list[list[Piece | None]] = [[None] * 8 for _ in range(8)]
        self.turn: Color = Color.WHITE
        self.castling_rights: dict[str, bool] = {"wQ": True, "wK": True, "bQ": True, "bK": True}
        self.en_passant: Square | None = None
        self.halfmove_clock: int = 0
        self.history: list[HistoryEntry] = []
        self.zobrist: int = 0

    @classmethod
    def starting_position(cls) -> "Board":
        board = cls()
        for col, kind in enumerate(BACK_RANK):
            board.grid[0][col] = Piece(Color.BLACK, kind)
            board.grid[7][col] = Piece(Color.WHITE, kind)
        for col in range(8):
            board.grid[1][col] = Piece(Color.BLACK, Kind.PAWN)
            board.grid[6][col] = Piece(Color.WHITE, Kind.PAWN)
        board.recompute_zobrist()
        return board

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        parts = fen.split()
        if len(parts) != 6:
            raise ValueError("Invalid FEN string")
        board_part, turn_part, castling_part, en_passant_part, halfmove_part, fullmove_part = parts
        board = cls()
        rows = board_part.split("/")
        if len(rows) != 8:
            raise ValueError("Invalid FEN string")
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    color = Color.WHITE if char.isupper() else Color.BLACK
                    kind = Kind(char.upper())
                    board.grid[row_idx][col_idx] = Piece(color, kind)
                    col_idx += 1
            if col_idx != 8:
                raise ValueError("Invalid FEN string")
        board.turn = Color.WHITE if turn_part == "w" else Color.BLACK
        board.castling_rights = {
            "wK": "K" in castling_part,
            "wQ": "Q" in castling_part,
            "bK": "k" in castling_part,
            "bQ": "q" in castling_part,
        }
        if en_passant_part == "-":
            board.en_passant = None
        else:
            col = ord(en_passant_part[0]) - ord("a")
            row = 8 - int(en_passant_part[1])
            board.en_passant = (row, col)
        board.halfmove_clock = int(halfmove_part)
        board.recompute_zobrist()
        return board

    def piece_at(self, square: Square) -> Piece | None:
        row, col = square
        return self.grid[row][col]

    def compute_zobrist(self) -> int:
        key = 0
        for row in range(8):
            grid_row = self.grid[row]
            for col in range(8):
                piece = grid_row[col]
                if piece is not None:
                    key ^= ZOBRIST_PIECES[piece.color][piece.kind][row * 8 + col]
        if self.turn is Color.BLACK:
            key ^= ZOBRIST_SIDE
        key ^= ZOBRIST_CASTLING[_castling_mask(self.castling_rights)]
        if self.en_passant is not None:
            key ^= ZOBRIST_EN_PASSANT[self.en_passant[1]]
        return key

    def recompute_zobrist(self) -> None:
        self.zobrist = self.compute_zobrist()

    def make_move(self, move: Move) -> None:
        start_row, start_col = move.start
        end_row, end_col = move.end
        grid = self.grid
        piece = grid[start_row][start_col]
        if piece is None:
            raise ValueError(f"No piece at starting square {move.start}")
        rights = self.castling_rights
        key = self.zobrist
        self.history.append(
            HistoryEntry(move, piece, dict(rights), self.en_passant, self.halfmove_clock, key)
        )
        color = piece.color
        kind = piece.kind
        own_keys = ZOBRIST_PIECES[color]
        flag = move.flag
        start_index = start_row * 8 + start_col
        end_index = end_row * 8 + end_col
        key ^= own_keys[kind][start_index]
        if flag == "en_passant":
            victim = grid[start_row][end_col]
            victim_index = start_row * 8 + end_col
            grid[start_row][end_col] = None
        else:
            victim = grid[end_row][end_col]
            victim_index = end_index
        if victim is not None:
            key ^= ZOBRIST_PIECES[victim.color][victim.kind][victim_index]
        placed = piece if move.promotion is None else Piece(color, move.promotion)
        grid[end_row][end_col] = placed
        key ^= own_keys[placed.kind][end_index]
        if flag == "castle_k":
            grid[start_row][5] = grid[start_row][7]
            grid[start_row][7] = None
            rook_keys = own_keys[Kind.ROOK]
            key ^= rook_keys[start_row * 8 + 7] ^ rook_keys[start_row * 8 + 5]
        elif flag == "castle_q":
            grid[start_row][3] = grid[start_row][0]
            grid[start_row][0] = None
            rook_keys = own_keys[Kind.ROOK]
            key ^= rook_keys[start_row * 8] ^ rook_keys[start_row * 8 + 3]
        grid[start_row][start_col] = None
        if kind is Kind.KING or move.start in CORNER_RIGHTS or move.end in CORNER_RIGHTS:
            old_mask = _castling_mask(rights)
            if kind is Kind.KING:
                rights[f"{color.value}K"] = False
                rights[f"{color.value}Q"] = False
            right = CORNER_RIGHTS.get(move.start)
            if right is not None:
                rights[right] = False
            right = CORNER_RIGHTS.get(move.end)
            if right is not None:
                rights[right] = False
            new_mask = _castling_mask(rights)
            if new_mask != old_mask:
                key ^= ZOBRIST_CASTLING[old_mask] ^ ZOBRIST_CASTLING[new_mask]
        if self.en_passant is not None:
            key ^= ZOBRIST_EN_PASSANT[self.en_passant[1]]
        if kind is Kind.PAWN and abs(end_row - start_row) == 2:
            self.en_passant = ((start_row + end_row) // 2, start_col)
            key ^= ZOBRIST_EN_PASSANT[start_col]
        else:
            self.en_passant = None
        if kind is Kind.PAWN or move.captured is not None:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        self.turn = self.turn.opponent()
        self.zobrist = key ^ ZOBRIST_SIDE

    def undo_move(self) -> None:
        if not self.history:
            raise ValueError("No moves to undo")
        entry = self.history.pop()
        move = entry.move
        start_row, start_col = move.start
        end_row, end_col = move.end
        self.grid[start_row][start_col] = entry.moved_piece
        if move.flag == "en_passant":
            self.grid[end_row][end_col] = None
            self.grid[start_row][end_col] = move.captured
        else:
            self.grid[end_row][end_col] = move.captured
        if move.flag == "castle_k":
            self.grid[start_row][7] = self.grid[start_row][5]
            self.grid[start_row][5] = None
        elif move.flag == "castle_q":
            self.grid[start_row][0] = self.grid[start_row][3]
            self.grid[start_row][3] = None
        self.castling_rights = dict(entry.castling_rights)
        self.en_passant = entry.en_passant
        self.halfmove_clock = entry.halfmove_clock
        self.zobrist = entry.zobrist
        self.turn = self.turn.opponent()

    def make_null_move(self) -> tuple[Square | None, int, int]:
        saved = (self.en_passant, self.zobrist, self.halfmove_clock)
        key = self.zobrist ^ ZOBRIST_SIDE
        if self.en_passant is not None:
            key ^= ZOBRIST_EN_PASSANT[self.en_passant[1]]
        self.en_passant = None
        self.zobrist = key
        self.halfmove_clock = 0
        self.turn = self.turn.opponent()
        return saved

    def undo_null_move(self, saved: tuple[Square | None, int, int]) -> None:
        self.en_passant, self.zobrist, self.halfmove_clock = saved
        self.turn = self.turn.opponent()

    def position_key(self) -> tuple:
        return (
            tuple(tuple(row) for row in self.grid),
            self.turn,
            tuple(sorted(self.castling_rights.items())),
            self.en_passant,
        )

    def to_fen(self, fullmove: int = 1) -> str:
        rows = []
        for grid_row in self.grid:
            text = ""
            empty = 0
            for piece in grid_row:
                if piece is None:
                    empty += 1
                    continue
                if empty:
                    text += str(empty)
                    empty = 0
                text += piece.kind.value if piece.color is Color.WHITE else piece.kind.value.lower()
            if empty:
                text += str(empty)
            rows.append(text)
        castling = "".join(
            letter
            for letter, right in (("K", "wK"), ("Q", "wQ"), ("k", "bK"), ("q", "bQ"))
            if self.castling_rights[right]
        )
        en_passant = from_square(self.en_passant) if self.en_passant is not None else "-"
        return " ".join(
            [
                "/".join(rows),
                self.turn.value,
                castling or "-",
                en_passant,
                str(self.halfmove_clock),
                str(fullmove),
            ]
        )

    def copy(self) -> "Board":
        new_board = Board()
        new_board.grid = [row.copy() for row in self.grid]
        new_board.turn = self.turn
        new_board.castling_rights = dict(self.castling_rights)
        new_board.en_passant = self.en_passant
        new_board.halfmove_clock = self.halfmove_clock
        new_board.history = self.history.copy()
        new_board.zobrist = self.zobrist
        return new_board

    def __str__(self) -> str:
        rows = []
        for idx, row in enumerate(self.grid):
            row_str = f"{8 - idx} "
            for piece in row:
                if piece is None:
                    row_str += "."
                else:
                    row_str += piece.kind.value.lower() if piece.color == Color.BLACK else piece.kind.value
            rows.append(row_str)
        rows.append("  abcdefgh")
        return "\n".join(rows)