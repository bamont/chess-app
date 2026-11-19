import argparse
import datetime
import math
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from bot.factory import create_bot
from chess.game import GameStatus
from chess.pgn import format_pgn, result_token, uci_moves_to_san
from tools.match import play_game_record, white_score
from tools.stats import ELO_SCALE, Z_95, color_summary, elo_bound, expected_score, wilson_interval
from tools.uci_bot import UciBot
from tools.uci_engine import UciEngine


@dataclass(frozen=True)
class LevelResult:
    level: int
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def score(self) -> float:
        return self.points / self.games


@dataclass(frozen=True)
class RatedGameTask:
    level: int
    index: int
    spec: str
    engine_command: tuple[str, ...]
    movetime_ms: int
    max_plies: int
    opening_plies: int
    seed: int


@dataclass(frozen=True)
class RatedGameResult:
    level: int
    index: int
    bot_is_white: bool
    status: GameStatus
    plies: int
    points: float
    moves: tuple[str, ...] = ()


def estimate_rating(results: list[LevelResult]) -> float | None:
    total_games = sum(result.games for result in results)
    total_points = sum(result.points for result in results)
    if total_games == 0 or total_points <= 0 or total_points >= total_games:
        return None

    def excess(rating: float) -> float:
        return sum(
            result.games * expected_score(rating - result.level) - result.points for result in results
        )

    low, high = -2000.0, 6000.0
    for _ in range(100):
        middle = (low + high) / 2
        if excess(middle) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def rating_standard_error(results: list[LevelResult], rating: float) -> float:
    information = 0.0
    for result in results:
        expected = expected_score(rating - result.level)
        information += result.games * expected * (1 - expected) * ELO_SCALE**2
    return 1 / math.sqrt(information)


def tally(level: int, results: list[RatedGameResult]) -> LevelResult:
    wins = sum(1 for result in results if result.points == 1)
    losses = sum(1 for result in results if result.points == 0)
    return LevelResult(level, wins, len(results) - wins - losses, losses)


def play_rated_game(task: RatedGameTask) -> RatedGameResult:
    bot_is_white = task.index % 2 == 0
    bot = create_bot(task.spec, seed=task.seed + 2 * task.index)
    with UciEngine(list(task.engine_command)) as engine:
        engine.set_option("UCI_LimitStrength", "true")
        engine.set_option("UCI_Elo", str(task.level))
        engine.new_game()
        opponent = UciBot(engine, task.movetime_ms)
        white, black = (bot, opponent) if bot_is_white else (opponent, bot)
        status, moves = play_game_record(
            white, black, task.max_plies, task.opening_plies, task.seed + task.index // 2
        )
    white_points = white_score(status)
    points = white_points if bot_is_white else 1 - white_points
    return RatedGameResult(task.level, task.index, bot_is_white, status, len(moves), points, tuple(moves))


def play_rated_games(tasks: list[RatedGameTask], workers: int) -> Iterator[RatedGameResult]:
    if workers <= 1:
        for task in tasks:
            yield play_rated_game(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(play_rated_game, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def write_pgn(results: list[RatedGameResult], bot_spec: str, path: str) -> None:
    date = datetime.date.today().strftime("%Y.%m.%d")
    games = []
    for result in sorted(results, key=lambda item: (item.level, item.index)):
        opponent = f"Stockfish UCI_Elo {result.level}"
        white, black = (bot_spec, opponent) if result.bot_is_white else (opponent, bot_spec)
        token = result_token(result.status)
        headers = {
            "Event": f"Rating test vs {opponent}",
            "Site": "?",
            "Date": date,
            "Round": str(result.index + 1),
            "White": white,
            "Black": black,
            "Result": token,
            "Termination": result.status.value,
        }
        games.append(format_pgn(headers, uci_moves_to_san(result.moves), token))
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(games))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a bot's rating against a UCI engine at limited strength.")
    parser.add_argument("bot", help="bot specification, for example minimax:3")
    parser.add_argument("--engine", required=True, help="path to a UCI engine such as Stockfish")
    parser.add_argument("--levels", type=int, nargs="+", default=[1320], help="UCI_Elo levels to play against")
    parser.add_argument("--games", type=int, default=10, help="games per level")
    parser.add_argument("--movetime", type=int, default=200, help="engine time per move in milliseconds")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="games played in parallel, one engine process each")
    parser.add_argument("--pgn", help="write every game to this PGN file")
    args = parser.parse_args()
    try:
        create_bot(args.bot)
    except ValueError as error:
        parser.error(str(error))

    with UciEngine([args.engine]) as probe:
        elo_range = probe.option_range("UCI_Elo")
        if elo_range is None or not probe.has_option("UCI_LimitStrength"):
            parser.error("This engine does not support UCI_LimitStrength and UCI_Elo")

    levels: list[int] = []
    for requested in args.levels:
        level = min(max(requested, elo_range[0]), elo_range[1])
        if level != requested:
            print(f"Level {requested} is outside the engine range {elo_range}, using {level}")
        if level not in levels:
            levels.append(level)

    tasks = [
        RatedGameTask(
            level,
            index,
            args.bot,
            (args.engine,),
            args.movetime,
            args.max_plies,
            args.opening_plies,
            args.seed,
        )
        for level in levels
        for index in range(args.games)
    ]
    workers = max(1, min(args.workers, len(tasks)))
    start = time.perf_counter()
    results: list[RatedGameResult] = []
    for result in play_rated_games(tasks, workers):
        results.append(result)
        side = "white" if result.bot_is_white else "black"
        print(
            f"  level {result.level} game {result.index + 1}: bot plays {side}, "
            f"{result.status.value}, {result.plies} plies",
            flush=True,
        )

    level_results = [tally(level, [r for r in results if r.level == level]) for level in levels]
    print()
    for result in level_results:
        low, high = wilson_interval(result.score, result.games)
        print(
            f"level {result.level}: +{result.wins} ={result.draws} -{result.losses}, "
            f"score {result.score:.2f}, Elo difference [{elo_bound(low)}, {elo_bound(high)}]"
        )
    print(color_summary((result.bot_is_white, result.points) for result in results))
    print(f"{len(results)} games in {time.perf_counter() - start:.0f}s with {workers} worker(s)")
    if args.pgn:
        write_pgn(results, args.bot, args.pgn)
        print(f"games written to {args.pgn}")
    estimate = estimate_rating(level_results)
    if estimate is None:
        print("No estimate: the bot won or lost every game. Try other levels, a weaker or a stronger opponent.")
        return
    margin = Z_95 * rating_standard_error(level_results, estimate)
    print(f"Estimated rating: {estimate:.0f} +/- {margin:.0f} (95%)")
    print("This is on the engine's UCI_Elo scale, not a Lichess, Chess.com or FIDE rating.")


if __name__ == "__main__":
    main()