import sys
from pathlib import Path

from chess.game import GameStatus
from tools.rating import RatedGameResult, RatedGameTask, play_rated_game, play_rated_games, tally

FAKE_ENGINE = (sys.executable, str(Path(__file__).parent / "fake_uci_engine.py"))


def task(level: int = 1500, index: int = 0, seed: int = 1) -> RatedGameTask:
    return RatedGameTask(level, index, "random", FAKE_ENGINE, 10, 20, 2, seed)


def key(result: RatedGameResult):
    return (result.level, result.index, result.bot_is_white, result.status, result.plies, result.points)


def test_game_result_is_consistent():
    result = play_rated_game(task())
    assert result.level == 1500
    assert result.bot_is_white
    assert result.plies <= 20
    assert isinstance(result.status, GameStatus)
    assert result.points in (0.0, 0.5, 1.0)


def test_colors_alternate_with_the_game_index():
    assert play_rated_game(task(index=0)).bot_is_white
    assert not play_rated_game(task(index=1)).bot_is_white


def test_a_game_is_reproducible_with_a_deterministic_engine():
    assert key(play_rated_game(task(index=2))) == key(play_rated_game(task(index=2)))


def test_tally_counts_wins_draws_and_losses():
    def result(points: float) -> RatedGameResult:
        return RatedGameResult(1500, 0, True, GameStatus.ONGOING, 10, points)

    level_result = tally(1500, [result(1.0), result(0.5), result(0.0), result(1.0)])
    assert (level_result.wins, level_result.draws, level_result.losses) == (2, 1, 1)
    assert level_result.level == 1500


def test_parallel_and_sequential_runs_give_the_same_results():
    tasks = [task(level, index) for level in (1400, 1600) for index in range(2)]
    sequential = sorted(key(r) for r in play_rated_games(tasks, workers=1))
    parallel = sorted(key(r) for r in play_rated_games(tasks, workers=2))
    assert sequential == parallel


def test_rated_games_keep_their_moves():
    from chess.pgn import uci_moves_to_san

    result = play_rated_game(task())
    assert len(result.moves) == result.plies
    assert len(uci_moves_to_san(result.moves)) == result.plies