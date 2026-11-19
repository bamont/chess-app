import argparse
import time

from bot.evaluation import evaluate_positional
from bot.minimax_bot import MinimaxBot
from chess.board import Board

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def run(depth: int, pruning: bool, quiescence: int, board: Board) -> None:
    bot = MinimaxBot(
        depth=depth,
        seed=0,
        pruning=pruning,
        evaluator=evaluate_positional,
        quiescence_limit=quiescence,
    )
    start = time.perf_counter()
    bot.search(board)
    elapsed = time.perf_counter() - start
    label = "alpha-beta" if pruning else "plain"
    print(f"depth {depth} {label:<10} {elapsed:7.2f}s {bot.nodes:>9} nodes {bot.qnodes:>9} qnodes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time the minimax bot.")
    parser.add_argument("--fen", default=START_FEN)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--plain-max-depth", type=int, default=2)
    parser.add_argument("--quiescence", type=int, default=6, help="0 disables quiescence search")
    args = parser.parse_args()
    board = Board.from_fen(args.fen)
    for depth in range(1, args.max_depth + 1):
        if depth <= args.plain_max_depth:
            run(depth, False, args.quiescence, board)
        run(depth, True, args.quiescence, board)


if __name__ == "__main__":
    main()