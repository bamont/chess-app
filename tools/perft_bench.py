import argparse

from chess.board import Board
from chess.rules import generate_legal_moves
from tools.timing import best_of, calibrate

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION_3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION_4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION_5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POSITION_6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

REFERENCE = {
    "start": (START, {1: 20, 2: 400, 3: 8902, 4: 197281}),
    "kiwipete": (KIWIPETE, {1: 48, 2: 2039, 3: 97862}),
    "position3": (POSITION_3, {1: 14, 2: 191, 3: 2812, 4: 43238}),
    "position4": (POSITION_4, {1: 6, 2: 264, 3: 9467}),
    "position5": (POSITION_5, {1: 44, 2: 1486, 3: 62379}),
    "position6": (POSITION_6, {1: 46, 2: 2079, 3: 89890}),
}


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    count = 0
    for move in generate_legal_moves(board, board.turn):
        board.make_move(move)
        count += perft(board, depth - 1)
        board.undo_move()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark and check the legal move generator.")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3, help="runs per position, the fastest is kept")
    parser.add_argument("--positions", nargs="+", choices=sorted(REFERENCE), default=list(REFERENCE))
    args = parser.parse_args()

    calibration = calibrate()
    print(f"machine calibration: {calibration:.3f}s (lower is faster); norm = nodes/s x calibration")
    failed = False
    total_nodes = 0
    total_time = 0.0
    for name in args.positions:
        fen, expected_by_depth = REFERENCE[name]
        expected = expected_by_depth.get(args.depth)
        counted: list[int] = []

        def run(fen: str = fen, counted: list[int] = counted) -> None:
            counted.append(perft(Board.from_fen(fen), args.depth))

        elapsed = max(best_of(run, args.repeat), 1e-9)
        nodes = counted[-1]
        if expected is None:
            verdict = "no reference"
        elif nodes == expected:
            verdict = "ok"
        else:
            verdict = f"FAIL (expected {expected})"
            failed = True
        speed = nodes / elapsed
        total_nodes += nodes
        total_time += elapsed
        print(
            f"{name:<10} depth {args.depth} {nodes:>9} nodes {elapsed:7.2f}s "
            f"{speed:>9.0f} nodes/s {speed * calibration:>7.0f} norm  {verdict}"
        )
    total_speed = total_nodes / max(total_time, 1e-9)
    print(
        f"{'total':<10} {'':<7} {total_nodes:>9} nodes {total_time:7.2f}s "
        f"{total_speed:>9.0f} nodes/s {total_speed * calibration:>7.0f} norm"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()