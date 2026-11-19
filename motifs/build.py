import argparse
import csv
import sys
from collections.abc import Iterator

from motifs.dataset import THEMES, BuildReport, build_dataset, save_dataset


def read_records(path: str) -> Iterator[dict[str, str]]:
    csv.field_size_limit(2**31 - 1)
    if path == "-":
        sys.stdin.reconfigure(encoding="utf-8", newline="")
        yield from csv.DictReader(sys.stdin)
        return
    with open(path, encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def format_report(report: BuildReport, top_other: int = 15) -> str:
    lines = [f"read {report.read} puzzles, kept {report.kept}, skipped {report.skipped}"]
    if report.negatives:
        lines.append(f"added {report.negatives} positions from before the first move, without any label")
    for name, count in sorted(report.theme_counts.items(), key=lambda item: -item[1]):
        lines.append(f"  {name}: {count}")
    if report.unseen_themes:
        lines.append("themes never seen (check the spelling): " + ", ".join(report.unseen_themes))
    ranked = sorted(report.unknown_theme_counts.items(), key=lambda item: -item[1])[:top_other]
    if ranked:
        lines.append("most frequent themes not used as labels: " + ", ".join(f"{n} ({c})" for n, c in ranked))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn the Lichess puzzle CSV into a training set of positions and motif labels.")
    parser.add_argument("csv", help="lichess_db_puzzle.csv, or - to read it from standard input (for example from zstd -dc)")
    parser.add_argument("output", help="compressed .npz file to write")
    parser.add_argument("--limit", type=int, help="only read the first puzzles")
    parser.add_argument("--themes", nargs="+", default=list(THEMES), help="themes to use as labels")
    parser.add_argument(
        "--with-negatives",
        action="store_true",
        help="also add the position before the first move of every puzzle, with no theme (used to calibrate thresholds)",
    )
    args = parser.parse_args()
    dataset, report = build_dataset(read_records(args.csv), args.themes, args.limit, args.with_negatives)
    if report.kept == 0:
        parser.error("No puzzle could be read: is this the Lichess puzzle CSV?")
    save_dataset(args.output, dataset)
    print(format_report(report))
    print(f"dataset written to {args.output}")


if __name__ == "__main__":
    main()