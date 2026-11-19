from bot.random_bot import RandomBot
from chess.game import GameStatus
from chess.pgn import format_pgn, result_token, uci_moves_to_san
from tools.match import GameResult, GameTask, play_game, play_game_record, run_game


def test_result_tokens():
    assert result_token(GameStatus.WHITE_WINS) == "1-0"
    assert result_token(GameStatus.BLACK_WINS) == "0-1"
    assert result_token(GameStatus.ONGOING) == "*"
    assert result_token(GameStatus.STALEMATE) == "1/2-1/2"
    assert result_token(GameStatus.DRAW_REPETITION) == "1/2-1/2"


def test_pgn_layout():
    headers = {"Event": "Test", "White": "A", "Black": "B", "Result": "1-0"}
    text = format_pgn(headers, ["e4", "e5", "Nf3"], "1-0")
    assert text == '[Event "Test"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 1-0\n'


def test_quotes_in_headers_are_escaped():
    text = format_pgn({"White": 'say "hi"'}, [], "*")
    assert '[White "say \\"hi\\""]' in text


def test_long_games_wrap_at_80_columns():
    moves = ["Nf3", "Nf6", "Ng1", "Ng8"] * 20
    text = format_pgn({"Event": "Long"}, moves, "1/2-1/2")
    movetext = text.split("\n\n", 1)[1].splitlines()
    assert len(movetext) > 1
    assert all(len(line) <= 80 for line in movetext)
    assert movetext[-1].endswith("1/2-1/2")


def test_a_recorded_game_replays_to_san():
    status, moves = play_game_record(RandomBot(seed=1), RandomBot(seed=2), 30, 2, 3)
    assert isinstance(status, GameStatus)
    assert len(uci_moves_to_san(moves)) == len(moves)


def test_play_game_and_play_game_record_agree():
    first = play_game(RandomBot(seed=4), RandomBot(seed=5), 40, 2, 7)
    status, moves = play_game_record(RandomBot(seed=4), RandomBot(seed=5), 40, 2, 7)
    assert first == (status, len(moves))


def test_match_results_keep_their_moves():
    result = run_game(GameTask(0, "random", "random", 20, 2, 1))
    assert len(result.moves) == result.plies
    assert len(uci_moves_to_san(result.moves)) == result.plies


def test_results_without_moves_default_to_empty():
    assert GameResult(0, True, GameStatus.ONGOING, 10, 0.5, 0.1).moves == ()