import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from bot.factory import create_bot
from chess.board import Board
from chess.notation import move_to_uci, parse_uci
from chess.rules import generate_legal_moves
from chess.san import move_to_san
from tools.analyze_games import centipawn_loss
from tools.uci_engine import MATE_CENTIPAWNS, UciEngine

DEFAULT_BOTS = ("minimax:3:nobook", "minimax:4:nobook", "minimax:5:nobook")
LINE_PLIES = 6
REFUTATION_PLIES = 8
MATE_SCORE_CP = MATE_CENTIPAWNS - 1000
REFUTATION_KINDS = ("mate", "capture", "check", "quiet")


@dataclass(frozen=True)
class TacticsTask:
    position_id: str
    fen: str
    specs: tuple[str, ...]
    depth: int
    engine_command: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class SpecOutcome:
    spec: str
    move_uci: str
    move_san: str
    loss: int
    seconds: float
    refutation: tuple[str, ...] = ()
    mated: bool = False


@dataclass(frozen=True)
class PositionResult:
    id: str
    fen: str
    best_san: str
    outcomes: tuple[SpecOutcome, ...]
    best_line: tuple[str, ...] = ()


def load_positions(path: str, limit: int | None = None) -> list[dict]:
    positions = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                positions.append(json.loads(line))
    return positions[:limit] if limit is not None else positions


def _san_of(fen: str, uci: str) -> str:
    if uci == "(none)":
        return "-"
    board = Board.from_fen(fen)
    return move_to_san(board, parse_uci(uci, generate_legal_moves(board, board.turn)))


def _line_san(fen: str, ucis: tuple[str, ...], limit: int = LINE_PLIES) -> tuple[str, ...]:
    board = Board.from_fen(fen)
    sans = []
    for uci in ucis[:limit]:
        try:
            move = parse_uci(uci, generate_legal_moves(board, board.turn))
        except ValueError:
            break
        sans.append(move_to_san(board, move))
        board.make_move(move)
    return tuple(sans)


def refutation_kind(outcome: SpecOutcome) -> str:
    if outcome.mated:
        return "mate"
    reply = outcome.refutation[1] if len(outcome.refutation) > 1 else ""
    if reply.endswith("#"):
        return "mate"
    if "x" in reply:
        return "capture"
    if reply.endswith("+"):
        return "check"
    return "quiet"


def run_position(task: TacticsTask) -> PositionResult:
    with UciEngine(list(task.engine_command)) as engine:
        engine.new_game()
        before = engine.analyse([], task.depth, fen=task.fen)
        outcomes = []
        for spec in task.specs:
            bot = create_bot(spec, seed=task.seed)
            board = Board.from_fen(task.fen)
            start = time.perf_counter()
            move = bot.choose_move(board)
            seconds = time.perf_counter() - start
            uci = move_to_uci(move)
            after = engine.analyse([uci], task.depth, fen=task.fen)
            outcomes.append(
                SpecOutcome(
                    spec,
                    uci,
                    _san_of(task.fen, uci),
                    centipawn_loss(before.score_cp, after.score_cp),
                    seconds,
                    _line_san(task.fen, (uci,) + after.pv, REFUTATION_PLIES),
                    after.score_cp >= MATE_SCORE_CP,
                )
            )
    return PositionResult(
        task.position_id,
        task.fen,
        _san_of(task.fen, before.best_move),
        tuple(outcomes),
        _line_san(task.fen, before.pv),
    )


def format_report(results: list[PositionResult], specs: tuple[str, ...], threshold: int) -> str:
    if not results:
        return "no positions"
    lines = [f"{len(results)} positions, a move is solved when it loses less than {threshold} cp"]
    for index, spec in enumerate(specs):
        outcomes = [result.outcomes[index] for result in results]
        solved = sum(1 for outcome in outcomes if outcome.loss < threshold)
        average_loss = sum(outcome.loss for outcome in outcomes) / len(outcomes)
        average_time = sum(outcome.seconds for outcome in outcomes) / len(outcomes)
        lines.append(
            f"{spec}: solved {solved}/{len(outcomes)} ({100 * solved / len(outcomes):.0f}%), "
            f"average loss {average_loss:.0f} cp, average time {average_time:.2f}s"
        )
    first_counts = [0] * len(specs)
    never = 0
    for result in results:
        index = next((i for i, outcome in enumerate(result.outcomes) if outcome.loss < threshold), None)
        if index is None:
            never += 1
        else:
            first_counts[index] += 1
    parts = [f"{spec} {count}" for spec, count in zip(specs, first_counts, strict=False)] + [f"none {never}"]
    lines.append("first solved by: " + ", ".join(parts))
    return "\n".join(lines)


def format_details(results: list[PositionResult]) -> str:
    lines = []
    for result in results:
        moves = ", ".join(f"{o.spec} {o.move_san} (loss {o.loss})" for o in result.outcomes)
        lines.append(f"{result.id}: best {result.best_san}; {moves}")
    return "\n".join(lines)


def format_unsolved(results: list[PositionResult], threshold: int) -> str:
    unsolved = [result for result in results if all(o.loss >= threshold for o in result.outcomes)]
    if not unsolved:
        return "every position is solved by at least one bot"
    kinds = Counter(refutation_kind(result.outcomes[-1]) for result in unsolved)
    lines = [
        f"{len(unsolved)} positions unsolved by every bot:",
        "  refutation of the last bot's move: " + ", ".join(f"{kind} {kinds[kind]}" for kind in REFUTATION_KINDS),
    ]
    for result in unsolved:
        line = " ".join(result.best_line) if result.best_line else result.best_san
        lines.append(f"  {result.id}: best line {line}")
        lines.append("    bots: " + ", ".join(f"{o.spec} {o.move_san} ({o.loss})" for o in result.outcomes))
        refutation = result.outcomes[-1].refutation
        if refutation:
            lines.append("    refutation: " + " ".join(refutation))
        lines.append(f"    {result.fen}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test bots on the blunder positions found by analyze_games.")
    parser.add_argument("positions", help="JSON Lines file written by tools.analyze_games --blunders-file")
    parser.add_argument("--engine", required=True, help="path to a UCI engine such as Stockfish")
    parser.add_argument("--bots", nargs="+", default=list(DEFAULT_BOTS), help="bot specifications to test")
    parser.add_argument("--depth", type=int, default=12, help="engine analysis depth")
    parser.add_argument("--threshold", type=int, default=100, help="loss in centipawns under which a move is solved")
    parser.add_argument("--limit", type=int, help="only use the first positions")
    parser.add_argument("--workers", type=int, default=1, help="positions processed in parallel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="print the move chosen on every position")
    parser.add_argument("--unsolved", action="store_true", help="list the positions no bot solves")
    args = parser.parse_args()
    for spec in args.bots:
        try:
            create_bot(spec)
        except ValueError as error:
            parser.error(str(error))

    specs = tuple(args.bots)
    tasks = [
        TacticsTask(record["id"], record["fen"], specs, args.depth, (args.engine,), args.seed)
        for record in load_positions(args.positions, args.limit)
    ]
    if not tasks:
        parser.error("No position to test")
    workers = max(1, min(args.workers, len(tasks)))
    start = time.perf_counter()
    if workers == 1:
        results = [run_position(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_position, tasks))
    if args.verbose:
        print(format_details(results))
    print(format_report(results, specs, args.threshold))
    if args.unsolved:
        print(format_unsolved(results, args.threshold))
    print(f"{time.perf_counter() - start:.0f}s with {workers} worker(s)")


if __name__ == "__main__":
    main()