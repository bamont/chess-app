import argparse
import cProfile
import io
import pstats
import time

from bot.factory import create_bot
from chess.board import Board
from tools.timing import calibrate

POSITIONS = {
    "start": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "middlegame": "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
}


def run_search(spec: str, fen: str) -> tuple[float, int]:
    bot = create_bot(spec, seed=0)
    board = Board.from_fen(fen)
    start = time.perf_counter()
    bot.choose_move(board)
    elapsed = time.perf_counter() - start
    nodes = getattr(bot, "nodes", 0) + getattr(bot, "qnodes", 0)
    return elapsed, nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Time and profile a bot's search.")
    parser.add_argument("--bot", default="minimax:3:nobook")
    parser.add_argument("--positions", nargs="+", choices=sorted(POSITIONS), default=sorted(POSITIONS))
    parser.add_argument("--repeat", type=int, default=3, help="runs per position, the fastest is kept")
    parser.add_argument("--profile", action="store_true", help="print cProfile statistics (single run)")
    parser.add_argument("--sort", choices=("tottime", "cumulative"), default="tottime")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    try:
        create_bot(args.bot)
    except ValueError as error:
        parser.error(str(error))

    calibration = calibrate()
    print(f"machine calibration: {calibration:.3f}s (lower is faster); norm = nodes/s x calibration")
    profiler = cProfile.Profile() if args.profile else None
    total_time = 0.0
    total_nodes = 0
    for name in args.positions:
        runs = 1 if profiler is not None else args.repeat
        best_elapsed = float("inf")
        nodes = 0
        for _ in range(runs):
            if profiler is not None:
                profiler.enable()
            elapsed, nodes = run_search(args.bot, POSITIONS[name])
            if profiler is not None:
                profiler.disable()
            best_elapsed = min(best_elapsed, elapsed)
        speed = nodes / max(best_elapsed, 1e-9)
        total_time += best_elapsed
        total_nodes += nodes
        print(f"{name:<12} {best_elapsed:7.2f}s {nodes:>9} nodes {speed:>9.0f} nodes/s {speed * calibration:>7.0f} norm")
    total_speed = total_nodes / max(total_time, 1e-9)
    print(f"{'total':<12} {total_time:7.2f}s {total_nodes:>9} nodes {total_speed:>9.0f} nodes/s {total_speed * calibration:>7.0f} norm")
    if profiler is not None:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats(args.sort).print_stats(args.top)
        print(stream.getvalue())


if __name__ == "__main__":
    main()