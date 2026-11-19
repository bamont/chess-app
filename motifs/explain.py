import argparse

from motifs.infer import MotifModel


def format_scores(model: MotifModel, fen: str, top: int) -> str:
    lines = [f"{'theme':<20}{'score':>8}{'threshold':>11}  selected"]
    for item in model.all_scores(fen)[:top]:
        mark = "yes" if item.score >= item.threshold else ""
        lines.append(f"{item.theme:<20}{item.score:>8.3f}{item.threshold:>11.3f}  {mark}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the motif scores of a position.")
    parser.add_argument("model", help=".npz file written by motifs.export")
    parser.add_argument("fen", help="position, with the side to move")
    parser.add_argument("--top", type=int, default=8, help="number of themes to show")
    args = parser.parse_args()
    print(format_scores(MotifModel.load(args.model), args.fen, args.top))


if __name__ == "__main__":
    main()