from chess.board import Board
from chess.pieces import Color, Kind, Piece

PIECE_VALUES = {
    Kind.PAWN: 100,
    Kind.KNIGHT: 320,
    Kind.BISHOP: 330,
    Kind.ROOK: 500,
    Kind.QUEEN: 900,
    Kind.KING: 0,
}
MATE_SCORE = 100_000

# Piece-square tables from Tomasz Michniewski's Simplified Evaluation Function
# (chessprogramming.org, CC BY-SA 3.0). White's point of view, rank 8 first.
PAWN_TABLE = (
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
)
KNIGHT_TABLE = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)
BISHOP_TABLE = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)
ROOK_TABLE = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
)
QUEEN_TABLE = (
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
)
KING_MIDDLEGAME_TABLE = (
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
)
KING_ENDGAME_TABLE = (
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
)
PIECE_SQUARE_TABLES = {
    Kind.PAWN: PAWN_TABLE,
    Kind.KNIGHT: KNIGHT_TABLE,
    Kind.BISHOP: BISHOP_TABLE,
    Kind.ROOK: ROOK_TABLE,
    Kind.QUEEN: QUEEN_TABLE,
}


def _mirror(table: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(table[(7 - row) * 8 + col] for row in range(8) for col in range(8))


WHITE_PIECE_TABLES = {
    kind: tuple(PIECE_VALUES[kind] + bonus for bonus in table)
    for kind, table in PIECE_SQUARE_TABLES.items()
}
BLACK_PIECE_TABLES = {kind: _mirror(table) for kind, table in WHITE_PIECE_TABLES.items()}
BLACK_KING_MIDDLEGAME_TABLE = _mirror(KING_MIDDLEGAME_TABLE)
BLACK_KING_ENDGAME_TABLE = _mirror(KING_ENDGAME_TABLE)


def evaluate(board: Board) -> int:
    score = 0
    turn = board.turn
    for row in board.grid:
        for piece in row:
            if piece is None:
                continue
            value = PIECE_VALUES[piece.kind]
            score += value if piece.color is turn else -value
    return score


def _endgame_from_counts(
    white_queens: int,
    white_rooks: int,
    white_minors: int,
    black_queens: int,
    black_rooks: int,
    black_minors: int,
) -> bool:
    if white_queens and (white_rooks or white_minors > 1):
        return False
    return not (black_queens and (black_rooks or black_minors > 1))


def is_endgame(board: Board) -> bool:
    counts = {color: [0, 0, 0] for color in Color}
    for row in board.grid:
        for piece in row:
            if piece is None:
                continue
            if piece.kind is Kind.QUEEN:
                counts[piece.color][0] += 1
            elif piece.kind is Kind.ROOK:
                counts[piece.color][1] += 1
            elif piece.kind is Kind.BISHOP or piece.kind is Kind.KNIGHT:
                counts[piece.color][2] += 1
    white, black = counts[Color.WHITE], counts[Color.BLACK]
    return _endgame_from_counts(*white, *black)


def piece_square_value(piece: Piece, row: int, col: int, endgame: bool) -> int:
    table_row = row if piece.color is Color.WHITE else 7 - row
    if piece.kind is Kind.KING:
        return (KING_ENDGAME_TABLE if endgame else KING_MIDDLEGAME_TABLE)[table_row * 8 + col]
    return PIECE_SQUARE_TABLES[piece.kind][table_row * 8 + col]


def evaluate_positional(board: Board) -> int:
    score = 0
    white_queens = white_rooks = white_minors = 0
    black_queens = black_rooks = black_minors = 0
    white_king = black_king = -1
    for row in range(8):
        grid_row = board.grid[row]
        base = row * 8
        for col in range(8):
            piece = grid_row[col]
            if piece is None:
                continue
            kind = piece.kind
            index = base + col
            if piece.color is Color.WHITE:
                if kind is Kind.KING:
                    white_king = index
                    continue
                score += WHITE_PIECE_TABLES[kind][index]
                if kind is Kind.QUEEN:
                    white_queens += 1
                elif kind is Kind.ROOK:
                    white_rooks += 1
                elif kind is Kind.BISHOP or kind is Kind.KNIGHT:
                    white_minors += 1
            else:
                if kind is Kind.KING:
                    black_king = index
                    continue
                score -= BLACK_PIECE_TABLES[kind][index]
                if kind is Kind.QUEEN:
                    black_queens += 1
                elif kind is Kind.ROOK:
                    black_rooks += 1
                elif kind is Kind.BISHOP or kind is Kind.KNIGHT:
                    black_minors += 1
    if _endgame_from_counts(white_queens, white_rooks, white_minors, black_queens, black_rooks, black_minors):
        white_king_table, black_king_table = KING_ENDGAME_TABLE, BLACK_KING_ENDGAME_TABLE
    else:
        white_king_table, black_king_table = KING_MIDDLEGAME_TABLE, BLACK_KING_MIDDLEGAME_TABLE
    if white_king >= 0:
        score += white_king_table[white_king]
    if black_king >= 0:
        score -= black_king_table[black_king]
    return score if board.turn is Color.WHITE else -score