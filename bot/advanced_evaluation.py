from collections.abc import Callable

from bot.evaluation import (
    BLACK_KING_ENDGAME_TABLE,
    BLACK_KING_MIDDLEGAME_TABLE,
    BLACK_PIECE_TABLES,
    KING_ENDGAME_TABLE,
    KING_MIDDLEGAME_TABLE,
    PIECE_VALUES,
    WHITE_PIECE_TABLES,
)
from chess.board import Board
from chess.pieces import Color, Kind

PHASE_TOTAL = 24
PHASE_WEIGHTS = {
    Kind.PAWN: 0,
    Kind.KNIGHT: 1,
    Kind.BISHOP: 1,
    Kind.ROOK: 2,
    Kind.QUEEN: 4,
    Kind.KING: 0,
}
PASSED_PAWN_MG = (0, 5, 10, 20, 35, 60)
PASSED_PAWN_EG = (0, 10, 25, 50, 90, 150)
BISHOP_PAIR_MG = 25
BISHOP_PAIR_EG = 45
DOUBLED_PAWN_MG = 10
DOUBLED_PAWN_EG = 20
ISOLATED_PAWN_MG = 10
ISOLATED_PAWN_EG = 15
MOP_UP_MIN_LEAD = 400
MOP_UP_MAX_PHASE = 8
MOP_UP_CENTER_WEIGHT = 10
MOP_UP_PROXIMITY_WEIGHT = 4
DRAWISH_LEAD = 400
DRAWISH_DIVISOR = 4
SHIELD_MISSING = 15
SHIELD_ADVANCED = 6
SHIELD_OPEN_FILE = 10
SHIELD_CENTER_FILES = (3, 4)


def _divide(value: int, divisor: int) -> int:
    if value >= 0:
        return value // divisor
    return -((-value) // divisor)


def _center_distance(index: int) -> int:
    row, col = divmod(index, 8)
    return max(3 - row, row - 4) + max(3 - col, col - 4)


def _king_distance(first: int, second: int) -> int:
    first_row, first_col = divmod(first, 8)
    second_row, second_col = divmod(second, 8)
    return abs(first_row - second_row) + abs(first_col - second_col)


def _is_passed_white(row: int, col: int, black_min_row: list[int]) -> bool:
    for file in range(max(col - 1, 0), min(col + 1, 7) + 1):
        if black_min_row[file] < row:
            return False
    return True


def _is_passed_black(row: int, col: int, white_max_row: list[int]) -> bool:
    for file in range(max(col - 1, 0), min(col + 1, 7) + 1):
        if white_max_row[file] > row:
            return False
    return True


def _shield_penalty_white(king: int, white_max_row: list[int]) -> int:
    row, col = divmod(king, 8)
    if col in SHIELD_CENTER_FILES:
        return 0
    penalty = 0
    for file in range(max(col - 1, 0), min(col + 1, 7) + 1):
        pawn_row = white_max_row[file]
        distance = row - pawn_row
        if pawn_row < 0 or distance <= 0 or distance >= 3:
            penalty += SHIELD_MISSING
            if file == col:
                penalty += SHIELD_OPEN_FILE
        elif distance == 2:
            penalty += SHIELD_ADVANCED
    return penalty


def _shield_penalty_black(king: int, black_min_row: list[int]) -> int:
    row, col = divmod(king, 8)
    if col in SHIELD_CENTER_FILES:
        return 0
    penalty = 0
    for file in range(max(col - 1, 0), min(col + 1, 7) + 1):
        pawn_row = black_min_row[file]
        distance = pawn_row - row
        if pawn_row > 7 or distance <= 0 or distance >= 3:
            penalty += SHIELD_MISSING
            if file == col:
                penalty += SHIELD_OPEN_FILE
        elif distance == 2:
            penalty += SHIELD_ADVANCED
    return penalty


def _build_evaluator(shield: bool) -> Callable[[Board], int]:
    def evaluate(board: Board) -> int:
        base = 0
        mg = 0
        eg = 0
        phase = 0
        white_material = black_material = 0
        white_bishops = black_bishops = 0
        white_pawns: list[tuple[int, int]] = []
        black_pawns: list[tuple[int, int]] = []
        white_king = black_king = -1
        for row in range(8):
            grid_row = board.grid[row]
            for col in range(8):
                piece = grid_row[col]
                if piece is None:
                    continue
                kind = piece.kind
                index = row * 8 + col
                if piece.color is Color.WHITE:
                    if kind is Kind.KING:
                        white_king = index
                        continue
                    base += WHITE_PIECE_TABLES[kind][index]
                    white_material += PIECE_VALUES[kind]
                    if kind is Kind.PAWN:
                        white_pawns.append((row, col))
                    elif kind is Kind.BISHOP:
                        white_bishops += 1
                else:
                    if kind is Kind.KING:
                        black_king = index
                        continue
                    base -= BLACK_PIECE_TABLES[kind][index]
                    black_material += PIECE_VALUES[kind]
                    if kind is Kind.PAWN:
                        black_pawns.append((row, col))
                    elif kind is Kind.BISHOP:
                        black_bishops += 1
                phase += PHASE_WEIGHTS[kind]
        phase = min(phase, PHASE_TOTAL)

        if white_king >= 0:
            mg += KING_MIDDLEGAME_TABLE[white_king]
            eg += KING_ENDGAME_TABLE[white_king]
        if black_king >= 0:
            mg -= BLACK_KING_MIDDLEGAME_TABLE[black_king]
            eg -= BLACK_KING_ENDGAME_TABLE[black_king]

        if white_bishops >= 2:
            mg += BISHOP_PAIR_MG
            eg += BISHOP_PAIR_EG
        if black_bishops >= 2:
            mg -= BISHOP_PAIR_MG
            eg -= BISHOP_PAIR_EG

        white_files = [0] * 8
        black_files = [0] * 8
        white_max_row = [-1] * 8
        black_min_row = [8] * 8
        for row, col in white_pawns:
            white_files[col] += 1
            if row > white_max_row[col]:
                white_max_row[col] = row
        for row, col in black_pawns:
            black_files[col] += 1
            if row < black_min_row[col]:
                black_min_row[col] = row
        for row, col in white_pawns:
            if _is_passed_white(row, col, black_min_row):
                advance = min(max(6 - row, 0), 5)
                mg += PASSED_PAWN_MG[advance]
                eg += PASSED_PAWN_EG[advance]
            if (col == 0 or white_files[col - 1] == 0) and (col == 7 or white_files[col + 1] == 0):
                mg -= ISOLATED_PAWN_MG
                eg -= ISOLATED_PAWN_EG
        for row, col in black_pawns:
            if _is_passed_black(row, col, white_max_row):
                advance = min(max(row - 1, 0), 5)
                mg -= PASSED_PAWN_MG[advance]
                eg -= PASSED_PAWN_EG[advance]
            if (col == 0 or black_files[col - 1] == 0) and (col == 7 or black_files[col + 1] == 0):
                mg += ISOLATED_PAWN_MG
                eg += ISOLATED_PAWN_EG
        for file in range(8):
            if white_files[file] > 1:
                extra = white_files[file] - 1
                mg -= DOUBLED_PAWN_MG * extra
                eg -= DOUBLED_PAWN_EG * extra
            if black_files[file] > 1:
                extra = black_files[file] - 1
                mg += DOUBLED_PAWN_MG * extra
                eg += DOUBLED_PAWN_EG * extra

        if shield:
            if white_king >= 0:
                mg -= _shield_penalty_white(white_king, white_max_row)
            if black_king >= 0:
                mg += _shield_penalty_black(black_king, black_min_row)

        score = base + _divide(mg * phase + eg * (PHASE_TOTAL - phase), PHASE_TOTAL)

        lead = white_material - black_material
        if white_king >= 0 and black_king >= 0 and phase <= MOP_UP_MAX_PHASE and abs(lead) >= MOP_UP_MIN_LEAD:
            strong, weak = (white_king, black_king) if lead > 0 else (black_king, white_king)
            bonus = MOP_UP_CENTER_WEIGHT * _center_distance(weak) + MOP_UP_PROXIMITY_WEIGHT * (
                14 - _king_distance(strong, weak)
            )
            score += bonus if lead > 0 else -bonus

        if (lead > 0 and not white_pawns and lead < DRAWISH_LEAD) or (
            lead < 0 and not black_pawns and -lead < DRAWISH_LEAD
        ):
            score = _divide(score, DRAWISH_DIVISOR)

        return score if board.turn is Color.WHITE else -score

    return evaluate


evaluate_advanced = _build_evaluator(shield=True)
evaluate_advanced_without_shield = _build_evaluator(shield=False)