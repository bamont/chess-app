import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from chess.board import Board
from chess.notation import move_to_uci, parse_uci
from chess.pgn import read_pgn
from chess.rules import generate_legal_moves
from chess.san import move_to_san, parse_san
from tools.uci_engine import UciEngine

SCORE_CLAMP = 1000
PHASES = (("opening", 0, 20), ("middlegame", 20, 60), ("late game", 60, 100_000))


@dataclass(frozen=True)
class AnalysisTask:
    game_number: int
    san_moves: tuple[str, ...]
    analysed_colors: tuple[bool, ...]
    skip_plies: int
    depth: int
    engine_command: tuple[str, ...]


@dataclass(frozen=True)
class MoveReport:
    game_number: int
    ply: int
    played_san: str
    played_uci: str
    best_san: str
    best_uci: str
    loss: int
    fen: str


def clamp(score: int) -> int:
    return max(-SCORE_CLAMP, min(SCORE_CLAMP, score))


def centipawn_loss(score_before: int, score_after_opponent_view: int) -> int:
    return max(0, clamp(score_before) + clamp(score_after_opponent_view))


def _best_san(fen: str, best_uci: str) -> str:
    if best_uci == "(none)":
        return "-"
    board = Board.from_fen(fen)
    move = parse_uci(best_uci, generate_legal_moves(board, board.turn))
    return move_to_san(board, move)


def analyse_game(task: AnalysisTask) -> list[MoveReport]:
    board = Board.starting_position()
    ucis: list[str] = []
    sans: list[str] = []
    fens: list[str] = []
    for text in task.san_moves:
        move = parse_san(board, text)
        fens.append(board.to_fen(len(ucis) // 2 + 1))
        sans.append(move_to_san(board, move))
        ucis.append(move_to_uci(move))
        board.make_move(move)
    total = len(ucis)
    analyses = {}
    with UciEngine(list(task.engine_command)) as engine:
        engine.new_game()
        for ply in range(task.skip_plies, total + 1):
            analyses[ply] = engine.analyse(ucis[:ply], task.depth)
    reports = []
    for ply in range(task.skip_plies, total):
        if (ply % 2 == 0) not in task.analysed_colors:
            continue
        before, after = analyses[ply], analyses[ply + 1]
        reports.append(
            MoveReport(
                task.game_number,
                ply,
                sans[ply],
                ucis[ply],
                _best_san(fens[ply], before.best_move),
                before.best_move,
                centipawn_loss(before.score_cp, after.score_cp),
                fens[ply],
            )
        )
    return reports


def format_summary(reports: list[MoveReport], games: int, blunder: int, mistake: int) -> str:
    if not reports:
        return "no analysed moves"
    average = sum(report.loss for report in reports) / len(reports)
    lines = [f"{games} games, {len(reports)} analysed moves, average centipawn loss {average:.0f}"]
    for name, start, end in PHASES:
        subset = [report for report in reports if start <= report.ply < end]
        if subset:
            mean = sum(report.loss for report in subset) / len(subset)
            bad = sum(1 for report in subset if report.loss >= blunder)
            lines.append(f"  {name}: {len(subset)} moves, average loss {mean:.0f}, blunders {bad}")
    blunders = [report for report in reports if report.loss >= blunder]
    mistakes = [report for report in reports if mistake <= report.loss < blunder]
    games_with_blunder = len({report.game_number for report in blunders})
    lines.append(
        f"blunders (>= {blunder} cp): {len(blunders)} in {games_with_blunder} games; "
        f"mistakes ({mistake}-{blunder - 1} cp): {len(mistakes)}"
    )
    first: dict[int, int] = {}
    for report in sorted(blunders, key=lambda item: (item.game_number, item.ply)):
        first.setdefault(report.game_number, report.ply // 2 + 1)
    if first:
        lines.append(
            f"first blunder of each game: median move {statistics.median(first.values()):.0f}, "
            f"earliest {min(first.values())}, latest {max(first.values())}"
        )
    return "\n".join(lines)


def format_worst(reports: list[MoveReport], count: int) -> str:
    lines = ["worst moves:"]
    for report in sorted(reports, key=lambda item: -item.loss)[:count]:
        number = report.ply // 2 + 1
        dots = "." if report.ply % 2 == 0 else "..."
        lines.append(
            f"  game {report.game_number}, move {number}{dots} {report.played_san} (loss {report.loss}), "
            f"best {report.best_san}\n    {report.fen}"
        )
    return "\n".join(lines)


def write_blunders(reports: list[MoveReport], threshold: int, path: str) -> int:
    blunders = [report for report in reports if report.loss >= threshold]
    with open(path, "w", encoding="utf-8") as file:
        for report in blunders:
            record = {
                "id": f"game{report.game_number}-ply{report.ply}",
                "fen": report.fen,
                "best_uci": report.best_uci,
                "best_san": report.best_san,
                "played_uci": report.played_uci,
                "played_san": report.played_san,
                "loss": report.loss,
            }
            file.write(json.dumps(record) + "\n")
    return len(blunders)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the weak moves of a bot in PGN games with a UCI engine.")
    parser.add_argument("pgn", help="PGN file written by tools.match or tools.rating")
    parser.add_argument("--player", required=True, help="part of the bot's name in the PGN headers, such as minimax")
    parser.add_argument("--engine", required=True, help="path to a UCI engine such as Stockfish")
    parser.add_argument("--depth", type=int, default=12, help="analysis depth")
    parser.add_argument("--skip-plies", type=int, default=4, help="ignore the first plies (random openings)")
    parser.add_argument("--blunder", type=int, default=200, help="loss in centipawns that counts as a blunder")
    parser.add_argument("--mistake", type=int, default=100, help="loss in centipawns that counts as a mistake")
    parser.add_argument("--worst", type=int, default=15, help="number of worst moves to list")
    parser.add_argument("--workers", type=int, default=1, help="games analysed in parallel")
    parser.add_argument("--blunders-file", help="write the blunders to this JSON Lines file")
    args = parser.parse_args()

    with open(args.pgn, encoding="utf-8") as file:
        games = read_pgn(file.read())
    tasks = []
    for number, game in enumerate(games, start=1):
        colors = tuple(
            is_white
            for is_white, key in ((True, "White"), (False, "Black"))
            if args.player in game.headers.get(key, "")
        )
        if colors:
            tasks.append(
                AnalysisTask(number, tuple(game.moves), colors, args.skip_plies, args.depth, (args.engine,))
            )
    if not tasks:
        parser.error(f"No game has a player whose name contains '{args.player}'")

    workers = max(1, min(args.workers, len(tasks)))
    reports: list[MoveReport] = []
    if workers == 1:
        for task in tasks:
            reports.extend(analyse_game(task))
            print(f"analysed game {task.game_number}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(analyse_game, tasks):
                reports.extend(result)
    print(format_summary(reports, len(tasks), args.blunder, args.mistake))
    if reports:
        print(format_worst(reports, args.worst))
    if args.blunders_file:
        count = write_blunders(reports, args.blunder, args.blunders_file)
        print(f"{count} blunders written to {args.blunders_file}")


if __name__ == "__main__":
    main()