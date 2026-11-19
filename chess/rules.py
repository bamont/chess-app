from chess.board import Board
from chess.move import Move, Square
from chess.pieces import Color, Kind, Piece

KNIGHT_OFFSETS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
KING_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ROOK_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
BISHOP_DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
QUEEN_DIRECTIONS = ROOK_DIRECTIONS + BISHOP_DIRECTIONS
PROMOTION_KINDS = (Kind.QUEEN, Kind.ROOK, Kind.BISHOP, Kind.KNIGHT)
ROOK_SLIDERS = (Kind.ROOK, Kind.QUEEN)
BISHOP_SLIDERS = (Kind.BISHOP, Kind.QUEEN)
CASTLING_SPECS = (
    ("K", 7, (5, 6), (4, 5, 6), 6, "castle_k"),
    ("Q", 0, (1, 2, 3), (4, 3, 2), 2, "castle_q"),
)


def _build_targets(offsets: list[tuple[int, int]]) -> tuple:
    return tuple(
        tuple(
            tuple(
                (row + d_row, col + d_col)
                for d_row, d_col in offsets
                if 0 <= row + d_row < 8 and 0 <= col + d_col < 8
            )
            for col in range(8)
        )
        for row in range(8)
    )


def _build_rays(directions: list[tuple[int, int]]) -> tuple:
    table = []
    for row in range(8):
        table_row = []
        for col in range(8):
            rays = []
            for d_row, d_col in directions:
                ray = []
                r, c = row + d_row, col + d_col
                while 0 <= r < 8 and 0 <= c < 8:
                    ray.append((r, c))
                    r += d_row
                    c += d_col
                if ray:
                    rays.append(tuple(ray))
            table_row.append(tuple(rays))
        table.append(tuple(table_row))
    return tuple(table)


def _build_pawn_sources(row_offset: int) -> tuple:
    return tuple(
        tuple(
            tuple(
                (row + row_offset, col + d_col)
                for d_col in (-1, 1)
                if 0 <= row + row_offset < 8 and 0 <= col + d_col < 8
            )
            for col in range(8)
        )
        for row in range(8)
    )


KNIGHT_TARGETS = _build_targets(KNIGHT_OFFSETS)
KING_TARGETS = _build_targets(KING_OFFSETS)
ROOK_RAYS = _build_rays(ROOK_DIRECTIONS)
BISHOP_RAYS = _build_rays(BISHOP_DIRECTIONS)
QUEEN_RAYS = _build_rays(QUEEN_DIRECTIONS)
WHITE_PAWN_SOURCES = _build_pawn_sources(1)
BLACK_PAWN_SOURCES = _build_pawn_sources(-1)


def _step_moves(grid: list, square: Square, targets: tuple, color: Color) -> list[Move]:
    moves = []
    for end in targets:
        target = grid[end[0]][end[1]]
        if target is None or target.color is not color:
            moves.append(Move(square, end, target))
    return moves


def _step_captures(grid: list, square: Square, targets: tuple, color: Color) -> list[Move]:
    moves = []
    for end in targets:
        target = grid[end[0]][end[1]]
        if target is not None and target.color is not color:
            moves.append(Move(square, end, target))
    return moves


def _sliding_moves(grid: list, square: Square, rays: tuple, color: Color) -> list[Move]:
    moves = []
    for ray in rays:
        for end in ray:
            target = grid[end[0]][end[1]]
            if target is None:
                moves.append(Move(square, end))
            else:
                if target.color is not color:
                    moves.append(Move(square, end, target))
                break
    return moves


def _sliding_captures(grid: list, square: Square, rays: tuple, color: Color) -> list[Move]:
    moves = []
    for ray in rays:
        for end in ray:
            target = grid[end[0]][end[1]]
            if target is not None:
                if target.color is not color:
                    moves.append(Move(square, end, target))
                break
    return moves


def knight_moves(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    return _step_moves(board.grid, square, KNIGHT_TARGETS[row][col], piece.color)


def knight_captures(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    return _step_captures(board.grid, square, KNIGHT_TARGETS[row][col], piece.color)


def king_moves(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    moves = _step_moves(board.grid, square, KING_TARGETS[row][col], piece.color)
    moves.extend(castling_moves(board, square))
    return moves


def king_captures(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    return _step_captures(board.grid, square, KING_TARGETS[row][col], piece.color)


def rook_moves(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    return _sliding_moves(board.grid, square, ROOK_RAYS[row][col], piece.color)


def rook_captures(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    return _sliding_captures(board.grid, square, ROOK_RAYS[row][col], piece.color)


def bishop_moves(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    return _sliding_moves(board.grid, square, BISHOP_RAYS[row][col], piece.color)


def bishop_captures(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    return _sliding_captures(board.grid, square, BISHOP_RAYS[row][col], piece.color)


def queen_moves(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    return _sliding_moves(board.grid, square, QUEEN_RAYS[row][col], piece.color)


def queen_captures(board: Board, square: Square) -> list[Move]:
    row, col = square
    piece = board.grid[row][col]
    if piece is None:
        return []
    return _sliding_captures(board.grid, square, QUEEN_RAYS[row][col], piece.color)


def pawn_move_variants(start: Square, end: Square, captured: Piece | None, color: Color) -> list[Move]:
    promotion_row = 0 if color is Color.WHITE else 7
    if end[0] == promotion_row:
        return [Move(start, end, captured, kind) for kind in PROMOTION_KINDS]
    return [Move(start, end, captured)]


def _pawn_diagonal_moves(board: Board, square: Square, color: Color, forward_row: int) -> list[Move]:
    row, col = square
    grid = board.grid
    forward = grid[forward_row]
    moves = []
    for capture_col in (col - 1, col + 1):
        if not 0 <= capture_col < 8:
            continue
        target = forward[capture_col]
        if target is not None:
            if target.color is not color:
                moves.extend(pawn_move_variants(square, (forward_row, capture_col), target, color))
        elif board.en_passant == (forward_row, capture_col):
            neighbor = grid[row][capture_col]
            if neighbor == Piece(color.opponent(), Kind.PAWN):
                moves.append(Move(square, (forward_row, capture_col), captured=neighbor, flag="en_passant"))
    return moves


def pawn_moves(board: Board, square: Square) -> list[Move]:
    row, col = square
    grid = board.grid
    piece = grid[row][col]
    if piece is None or piece.kind is not Kind.PAWN:
        return []
    color = piece.color
    white = color is Color.WHITE
    direction = -1 if white else 1
    forward_row = row + direction
    if not 0 <= forward_row < 8:
        return []
    moves = []
    if grid[forward_row][col] is None:
        moves.extend(pawn_move_variants(square, (forward_row, col), None, color))
        if row == (6 if white else 1) and grid[row + 2 * direction][col] is None:
            moves.append(Move(square, (row + 2 * direction, col)))
    moves.extend(_pawn_diagonal_moves(board, square, color, forward_row))
    return moves


def pawn_captures(board: Board, square: Square) -> list[Move]:
    row, col = square
    grid = board.grid
    piece = grid[row][col]
    if piece is None or piece.kind is not Kind.PAWN:
        return []
    color = piece.color
    white = color is Color.WHITE
    forward_row = row - 1 if white else row + 1
    if not 0 <= forward_row < 8:
        return []
    moves = []
    if forward_row == (0 if white else 7) and grid[forward_row][col] is None:
        moves.extend(pawn_move_variants(square, (forward_row, col), None, color))
    moves.extend(_pawn_diagonal_moves(board, square, color, forward_row))
    return moves


def castling_moves(board: Board, square: Square) -> list[Move]:
    piece = board.piece_at(square)
    if piece is None or piece.kind is not Kind.KING:
        return []
    home_row = 7 if piece.color is Color.WHITE else 0
    if square != (home_row, 4):
        return []
    opponent = piece.color.opponent()
    moves = []
    for side, rook_col, empty_cols, king_path_cols, king_end_col, flag in CASTLING_SPECS:
        if not board.castling_rights[f"{piece.color.value}{side}"]:
            continue
        if board.grid[home_row][rook_col] != Piece(piece.color, Kind.ROOK):
            continue
        if any(board.grid[home_row][col] is not None for col in empty_cols):
            continue
        if any(is_square_attacked(board, (home_row, col), opponent) for col in king_path_cols):
            continue
        moves.append(Move(square, (home_row, king_end_col), flag=flag))
    return moves


MOVE_GENERATORS = {
    Kind.PAWN: pawn_moves,
    Kind.KNIGHT: knight_moves,
    Kind.BISHOP: bishop_moves,
    Kind.ROOK: rook_moves,
    Kind.QUEEN: queen_moves,
    Kind.KING: king_moves,
}
CAPTURE_GENERATORS = {
    Kind.PAWN: pawn_captures,
    Kind.KNIGHT: knight_captures,
    Kind.BISHOP: bishop_captures,
    Kind.ROOK: rook_captures,
    Kind.QUEEN: queen_captures,
    Kind.KING: king_captures,
}


def _generate(board: Board, color: Color, generators: dict) -> list[Move]:
    moves = []
    grid = board.grid
    for row in range(8):
        grid_row = grid[row]
        for col in range(8):
            piece = grid_row[col]
            if piece is not None and piece.color is color:
                moves.extend(generators[piece.kind](board, (row, col)))
    return moves


def generate_pseudo_legal_moves(board: Board, color: Color) -> list[Move]:
    return _generate(board, color, MOVE_GENERATORS)


def generate_pseudo_legal_captures(board: Board, color: Color) -> list[Move]:
    return _generate(board, color, CAPTURE_GENERATORS)


def is_square_attacked(board: Board, square: Square, by_color: Color) -> bool:
    row, col = square
    grid = board.grid
    for r, c in KNIGHT_TARGETS[row][col]:
        piece = grid[r][c]
        if piece is not None and piece.color is by_color and piece.kind is Kind.KNIGHT:
            return True
    for r, c in KING_TARGETS[row][col]:
        piece = grid[r][c]
        if piece is not None and piece.color is by_color and piece.kind is Kind.KING:
            return True
    for ray in ROOK_RAYS[row][col]:
        for r, c in ray:
            piece = grid[r][c]
            if piece is not None:
                if piece.color is by_color and (piece.kind is Kind.ROOK or piece.kind is Kind.QUEEN):
                    return True
                break
    for ray in BISHOP_RAYS[row][col]:
        for r, c in ray:
            piece = grid[r][c]
            if piece is not None:
                if piece.color is by_color and (piece.kind is Kind.BISHOP or piece.kind is Kind.QUEEN):
                    return True
                break
    sources = WHITE_PAWN_SOURCES[row][col] if by_color is Color.WHITE else BLACK_PAWN_SOURCES[row][col]
    for r, c in sources:
        piece = grid[r][c]
        if piece is not None and piece.color is by_color and piece.kind is Kind.PAWN:
            return True
    return False


def find_king(board: Board, color: Color) -> Square:
    for row in range(8):
        for col in range(8):
            piece = board.grid[row][col]
            if piece is not None and piece.color is color and piece.kind is Kind.KING:
                return (row, col)
    raise ValueError(f"No king found for color {color}")


def is_in_check(board: Board, color: Color) -> bool:
    return is_square_attacked(board, find_king(board, color), color.opponent())


def pinned_squares(board: Board, color: Color, king_square: Square) -> set[Square]:
    pinned: set[Square] = set()
    row, col = king_square
    grid = board.grid
    for rays, sliders in ((ROOK_RAYS[row][col], ROOK_SLIDERS), (BISHOP_RAYS[row][col], BISHOP_SLIDERS)):
        for ray in rays:
            candidate = None
            for square in ray:
                piece = grid[square[0]][square[1]]
                if piece is None:
                    continue
                if candidate is None:
                    if piece.color is not color:
                        break
                    candidate = square
                else:
                    if piece.color is not color and piece.kind in sliders:
                        pinned.add(candidate)
                    break
    return pinned


def _legal_subset(board: Board, color: Color, moves: list[Move]) -> list[Move]:
    king_square = find_king(board, color)
    opponent = color.opponent()
    in_check = is_square_attacked(board, king_square, opponent)
    pinned = set() if in_check else pinned_squares(board, color, king_square)
    legal = []
    for move in moves:
        if not in_check and move.start != king_square and move.start not in pinned and move.flag != "en_passant":
            legal.append(move)
            continue
        king_after = move.end if move.start == king_square else king_square
        board.make_move(move)
        if not is_square_attacked(board, king_after, opponent):
            legal.append(move)
        board.undo_move()
    return legal


def generate_legal_moves(board: Board, color: Color) -> list[Move]:
    return _legal_subset(board, color, generate_pseudo_legal_moves(board, color))


def generate_legal_captures(board: Board, color: Color) -> list[Move]:
    return _legal_subset(board, color, generate_pseudo_legal_captures(board, color))


def is_checkmate(board: Board, color: Color) -> bool:
    return is_in_check(board, color) and not generate_legal_moves(board, color)


def is_stalemate(board: Board, color: Color) -> bool:
    return not is_in_check(board, color) and not generate_legal_moves(board, color)


def has_insufficient_material(board: Board) -> bool:
    knights = 0
    bishops = 0
    bishop_square_colors = set()
    for row in range(8):
        grid_row = board.grid[row]
        for col in range(8):
            piece = grid_row[col]
            if piece is None:
                continue
            kind = piece.kind
            if kind is Kind.KING:
                continue
            if kind is Kind.KNIGHT:
                knights += 1
            elif kind is Kind.BISHOP:
                bishops += 1
                bishop_square_colors.add((row + col) % 2)
            else:
                return False
    if knights + bishops <= 1:
        return True
    if knights > 0:
        return False
    return len(bishop_square_colors) == 1