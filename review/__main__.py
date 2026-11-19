import argparse
import sys

from chess.pgn import read_pgn
from review.analysis import analyse_game
from review.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an HTML review of a chess game with a UCI engine.")
    parser.add_argument("pgn", help="PGN file that contains the game")
    parser.add_argument("--engine", required=True, help="path to a UCI engine such as Stockfish")
    parser.add_argument("--game", type=int, default=1, help="number of the game in the PGN file (starting at 1)")
    parser.add_argument("--depth", type=int, default=14, help="analysis depth")
    parser.add_argument("--threads", type=int, default=1, help="engine threads, when the engine supports it")
    parser.add_argument("-o", "--output", default="review.html", help="HTML file to write")
    parser.add_argument(
        "--skip-plies",
        type=int,
        default=0,
        help="opening plies to leave unjudged (use 4 for games written by tools.match and tools.rating)",
    )
    parser.add_argument("--motifs", help="NumPy model written by motifs.export, to suggest tactical themes at key moments")
    args = parser.parse_args()

    with open(args.pgn, encoding="utf-8") as file:
        games = read_pgn(file.read())
    if not 1 <= args.game <= len(games):
        parser.error(f"The file contains {len(games)} game(s), game {args.game} does not exist")
    game = games[args.game - 1]
    if not game.moves:
        parser.error("The selected game has no moves")

    def progress(done: int, total: int) -> None:
        print(f"\ranalysing position {done}/{total}", end="", file=sys.stderr, flush=True)

    try:
        plies = analyse_game(game, (args.engine,), args.depth, args.threads, progress)
    except ValueError as error:
        parser.error(str(error))
    print(file=sys.stderr)
    motif_model = None
    if args.motifs:
        from motifs.infer import MotifModel
        motif_model = MotifModel.load(args.motifs)
    report = build_report(game.headers, game.result, plies, args.skip_plies, motif_model)
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(report)
    print(f"review written to {args.output}")


if __name__ == "__main__":
    main()