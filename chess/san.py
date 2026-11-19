from chess.board import Board
from chess.move import Move
from chess.pieces import Kind
from chess.rules import generate_legal_moves, is_in_check
from chess.utils import from_square


def _disambiguation(board: Board, move: Move) -> str:
    piece = board.piece_at(move.start)
    rivals = [
        other
        for other in generate_legal_moves(board, piece.color)
        if other.end == move.end and other.start != move.start and board.piece_at(other.start) == piece
    ]
    if not rivals:
        return ""
    square = from_square(move.start)
    if all(from_square(other.start)[0] != square[0] for other in rivals):
        return square[0]
    if all(from_square(other.start)[1] != square[1] for other in rivals):
        return square[1]
    return square


def move_to_san(board: Board, move: Move) -> str:
    piece = board.piece_at(move.start)
    if piece is None:
        raise ValueError(f"No piece on {from_square(move.start)}")
    if move.flag == "castle_k":
        san = "O-O"
    elif move.flag == "castle_q":
        san = "O-O-O"
    else:
        capture = "x" if move.captured is not None else ""
        destination = from_square(move.end)
        if piece.kind is Kind.PAWN:
            prefix = from_square(move.start)[0] if capture else ""
            san = f"{prefix}{capture}{destination}"
            if move.promotion is not None:
                san += f"={move.promotion.value}"
        else:
            san = f"{piece.kind.value}{_disambiguation(board, move)}{capture}{destination}"
    board.make_move(move)
    if is_in_check(board, board.turn):
        san += "#" if not generate_legal_moves(board, board.turn) else "+"
    board.undo_move()
    return san


def parse_san(board: Board, text: str) -> Move:
    wanted = text.strip().rstrip("+#!?").replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    legal = generate_legal_moves(board, board.turn)
    if wanted in ("O-O", "O-O-O"):
        flag = "castle_k" if wanted == "O-O" else "castle_q"
        candidates = [move for move in legal if move.flag == flag]
    else:
        destination = wanted.split("=")[0][-2:]
        candidates = [
            move
            for move in legal
            if from_square(move.end) == destination and move.flag not in ("castle_k", "castle_q")
        ]
    for move in candidates:
        if move_to_san(board, move).rstrip("+#") == wanted:
            return move
    raise ValueError(f"No legal move matches '{text}'")