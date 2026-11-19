from chess.game import GameStatus
from chess.pieces import Color

STATUS_MESSAGES = {
    GameStatus.WHITE_WINS: "Checkmate. White wins.",
    GameStatus.BLACK_WINS: "Checkmate. Black wins.",
    GameStatus.STALEMATE: "Draw by stalemate.",
    GameStatus.DRAW_FIFTY_MOVES: "Draw by the fifty-move rule.",
    GameStatus.DRAW_REPETITION: "Draw by threefold repetition.",
    GameStatus.DRAW_INSUFFICIENT_MATERIAL: "Draw by insufficient material.",
}


def describe_status(status: GameStatus, turn: Color, in_check: bool) -> str:
    if status != GameStatus.ONGOING:
        return STATUS_MESSAGES[status]
    side = "White" if turn == Color.WHITE else "Black"
    return f"{side} to move (check)" if in_check else f"{side} to move"


def describe_hint(human_color: Color | None) -> str:
    if human_color is None:
        return "N: new game"
    side = "White" if human_color == Color.WHITE else "Black"
    return f"You: {side} | N: new game"