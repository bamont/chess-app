from bot.base import Bot
from chess.game import Game, GameStatus
from chess.notation import move_to_uci, parse_uci
from chess.pieces import Color
from chess.rules import is_in_check
from ui.messages import STATUS_MESSAGES

HELP_TEXT = """Commands:
  e2e4    play a move (add q, r, b or n for a promotion, e.g. e7e8q)
  moves   list the legal moves
  help    show this help
  quit    leave the game"""


def main(bot: Bot | None = None, human_color: Color = Color.WHITE) -> None:
    game = Game()
    print(HELP_TEXT)
    while True:
        print()
        print(game.board)
        status = game.status()
        if status != GameStatus.ONGOING:
            print(STATUS_MESSAGES[status])
            return
        side = "White" if game.board.turn == Color.WHITE else "Black"
        check_notice = " (check)" if is_in_check(game.board, game.board.turn) else ""
        if bot is not None and game.board.turn != human_color:
            move = bot.choose_move(game.board)
            print(f"Bot plays {move_to_uci(move)}")
            game.play(move)
            continue
        try:
            text = input(f"{side} to move{check_notice} > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text in ("quit", "exit"):
            return
        if text == "help":
            print(HELP_TEXT)
            continue
        if bot is not None:
            side = "White" if human_color == Color.WHITE else "Black"
            print(f"You play {side}.")
        if text == "moves":
            print(" ".join(sorted(move_to_uci(move) for move in game.legal_moves())))
            continue
        try:
            game.play(parse_uci(text, game.legal_moves()))
        except ValueError as error:
            print(error)