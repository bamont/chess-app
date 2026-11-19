import argparse
import datetime
import os
import random
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from bot.base import Bot
from bot.factory import create_bot
from chess.game import Game, GameStatus
from chess.notation import move_to_uci
from chess.pgn import format_pgn, result_token, uci_moves_to_san
from chess.pieces import Color
from tools.stats import color_summary, elo_bound, elo_difference, wilson_interval


def white_score(status: GameStatus) -> float:
    if status == GameStatus.WHITE_WINS:
        return 1.0
    if status == GameStatus.BLACK_WINS:
        return 0.0
    return 0.5


def play_random_opening(game: Game, plies: int, seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(plies):
        moves = game.legal_moves()
        if not moves or game.status() != GameStatus.ONGOING:
            return
        game.play(rng.choice(moves))


def play_game_record(
    white: Bot,
    black: Bot,
    max_plies: int,
    opening_plies: int = 0,
    opening_seed: int = 0,
) -> tuple[GameStatus, list[str]]:
    game = Game()
    play_random_opening(game, opening_plies, opening_seed)
    while len(game.board.history) < max_plies:
        status = game.status()
        if status != GameStatus.ONGOING:
            break
        bot = white if game.board.turn == Color.WHITE else black
        game.play(bot.choose_move(game.board))
    return game.status(), [move_to_uci(entry.move) for entry in game.board.history]


def play_game(
    white: Bot,
    black: Bot,
    max_plies: int,
    opening_plies: int = 0,
    opening_seed: int = 0,
) -> tuple[GameStatus, int]:
    status, moves = play_game_record(white, black, max_plies, opening_plies, opening_seed)
    return status, len(moves)


@dataclass(frozen=True)
class GameTask:
    index: int
    spec_a: str
    spec_b: str
    max_plies: int
    opening_plies: int
    seed: int


@dataclass(frozen=True)
class GameResult:
    index: int
    a_is_white: bool
    status: GameStatus
    plies: int
    points_a: float
    elapsed: float
    moves: tuple[str, ...] = ()


def run_game(task: GameTask) -> GameResult:
    a_is_white = task.index % 2 == 0
    bot_a = create_bot(task.spec_a, seed=task.seed + 2 * task.index)
    bot_b = create_bot(task.spec_b, seed=task.seed + 2 * task.index + 1)
    white, black = (bot_a, bot_b) if a_is_white else (bot_b, bot_a)
    start = time.perf_counter()
    status, moves = play_game_record(
        white, black, task.max_plies, task.opening_plies, task.seed + task.index // 2
    )
    elapsed = time.perf_counter() - start
    white_points = white_score(status)
    points_a = white_points if a_is_white else 1 - white_points
    return GameResult(task.index, a_is_white, status, len(moves), points_a, elapsed, tuple(moves))


def make_tasks(
    spec_a: str,
    spec_b: str,
    games: int,
    max_plies: int,
    opening_plies: int,
    seed: int,
) -> list[GameTask]:
    return [GameTask(index, spec_a, spec_b, max_plies, opening_plies, seed) for index in range(games)]


def play_games(tasks: list[GameTask], workers: int) -> Iterator[GameResult]:
    if workers <= 1:
        for task in tasks:
            yield run_game(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_game, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def summarize(results: list[GameResult]) -> tuple[int, int, int]:
    wins = sum(1 for result in results if result.points_a == 1)
    losses = sum(1 for result in results if result.points_a == 0)
    return wins, len(results) - wins - losses, losses


def write_pgn(results: list[GameResult], spec_a: str, spec_b: str, path: str) -> None:
    date = datetime.date.today().strftime("%Y.%m.%d")
    games = []
    for result in sorted(results, key=lambda item: item.index):
        white, black = (spec_a, spec_b) if result.a_is_white else (spec_b, spec_a)
        token = result_token(result.status)
        headers = {
            "Event": "Bot match",
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
    parser = argparse.ArgumentParser(description="Play games between two bots.")
    parser.add_argument("bot_a", help="for example minimax:3 or greedy")
    parser.add_argument("bot_b", help="for example minimax:3:material")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="games played in parallel")
    parser.add_argument("--pgn", help="write every game to this PGN file")
    args = parser.parse_args()
    try:
        create_bot(args.bot_a)
        create_bot(args.bot_b)
    except ValueError as error:
        parser.error(str(error))

    tasks = make_tasks(args.bot_a, args.bot_b, args.games, args.max_plies, args.opening_plies, args.seed)
    workers = max(1, min(args.workers, args.games))
    start = time.perf_counter()
    results: list[GameResult] = []
    for result in play_games(tasks, workers):
        results.append(result)
        side = "white" if result.a_is_white else "black"
        print(
            f"game {result.index + 1}: A plays {side}, {result.status.value}, "
            f"{result.plies} plies, {result.elapsed:.1f}s",
            flush=True,
        )

    wins, draws, losses = summarize(results)
    score = (wins + 0.5 * draws) / len(results)
    elo = elo_difference(score)
    elo_text = f"{round(elo):+d}" if elo is not None else "n/a"
    low, high = wilson_interval(score, len(results))
    endings = Counter(result.status.value for result in results)
    print(f"A: {wins} wins, {draws} draws, {losses} losses (score {score:.2f}, Elo difference {elo_text})")
    print(f"95% interval on the Elo difference: [{elo_bound(low)}, {elo_bound(high)}]")
    print(color_summary((result.a_is_white, result.points_a) for result in results))
    print("endings: " + ", ".join(f"{name} {count}" for name, count in sorted(endings.items())))
    print(f"{len(results)} games in {time.perf_counter() - start:.0f}s with {workers} worker(s)")
    print("A and B are the first and second bot given on the command line.")
    if args.pgn:
        write_pgn(results, args.bot_a, args.bot_b, args.pgn)
        print(f"games written to {args.pgn}")


if __name__ == "__main__":
    main()