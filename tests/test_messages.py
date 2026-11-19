from chess.game import GameStatus
from chess.pieces import Color
from ui.messages import describe_hint, describe_status


def test_ongoing_messages():
    assert describe_status(GameStatus.ONGOING, Color.WHITE, False) == "White to move"
    assert describe_status(GameStatus.ONGOING, Color.BLACK, True) == "Black to move (check)"


def test_finished_games_ignore_the_check_flag():
    assert describe_status(GameStatus.WHITE_WINS, Color.BLACK, True) == "Checkmate. White wins."
    assert describe_status(GameStatus.STALEMATE, Color.BLACK, False) == "Draw by stalemate."


def test_hint_without_a_bot():
    assert describe_hint(None) == "N: new game"


def test_hint_shows_the_human_color():
    assert describe_hint(Color.WHITE) == "You: White | N: new game"
    assert describe_hint(Color.BLACK) == "You: Black | N: new game"