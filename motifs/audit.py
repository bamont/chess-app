import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from motifs.encoding import encode_position, parse_fen
from motifs.infer import MotifModel


@dataclass(frozen=True)
class AuditReport:
    positions: int
    any_theme: float
    theme_rates: dict[str, float]


def selection_rates(scores: np.ndarray, thresholds: np.ndarray, themes: Sequence[str]) -> AuditReport:
    selected = scores >= thresholds[np.newaxis, :]
    total = len(scores)
    if total == 0:
        return AuditReport(0, 0.0, {name: 0.0 for name in themes})
    rates = {name: float(selected[:, index].mean()) for index, name in enumerate(themes)}
    return AuditReport(total, float(selected.any(axis=1).mean()), rates)


def score_fens(model: MotifModel, fens: Sequence[str], batch_size: int = 256) -> np.ndarray:
    planes = []
    for fen in fens:
        rows, white_to_move = parse_fen(fen)
        planes.append(encode_position(rows, white_to_move))
    outputs = [model.scores(np.stack(planes[start : start + batch_size])) for start in range(0, len(planes), batch_size)]
    return np.concatenate(outputs) if outputs else np.zeros((0, len(model.themes)))


def audit_fens(model: MotifModel, fens: Sequence[str], batch_size: int = 256) -> AuditReport:
    return selection_rates(score_fens(model, fens, batch_size), model.thresholds, model.themes)


def select_examples(
    scores: np.ndarray,
    thresholds: np.ndarray,
    themes: Sequence[str],
    fens: Sequence[str],
    count: int,
    seed: int = 0,
) -> list[tuple[str, list[tuple[str, float]]]]:
    selected = scores >= thresholds[np.newaxis, :]
    candidates = np.flatnonzero(selected.any(axis=1))
    chosen = np.random.default_rng(seed).permutation(candidates)[:count]
    examples = []
    for position in chosen:
        found = [(themes[index], float(scores[position, index])) for index in np.flatnonzero(selected[position])]
        found.sort(key=lambda item: -item[1])
        examples.append((fens[position], found))
    return examples


def format_examples(examples: Sequence[tuple[str, list[tuple[str, float]]]]) -> str:
    lines = []
    for fen, found in examples:
        lines.append(fen)
        lines.append("  " + ", ".join(f"{theme} {score:.2f}" for theme, score in found))
    return "\n".join(lines)


def format_audit(report: AuditReport, top: int = 10) -> str:
    lines = [
        f"{report.positions} positions, at least one theme selected on {100 * report.any_theme:.1f}% of them",
        "most frequent themes:",
    ]
    ranked = sorted(report.theme_rates.items(), key=lambda item: -item[1])[:top]
    lines.extend(f"  {name:<20}{100 * rate:>6.1f}%" for name, rate in ranked)
    return "\n".join(lines)


def game_fens(moves: Sequence[str], first_ply: int) -> list[str]:
    from chess.board import Board
    from chess.san import parse_san

    board = Board.starting_position()
    fens = []
    for ply, text in enumerate(moves):
        if ply >= first_ply:
            fens.append(board.to_fen(ply // 2 + 1))
        board.make_move(parse_san(board, text))
    return fens


def main() -> None:
    from chess.pgn import read_pgn

    parser = argparse.ArgumentParser(description="Measure how often the motif model fires on ordinary game positions.")
    parser.add_argument("model", help=".npz file written by motifs.export")
    parser.add_argument("pgn", help="PGN file with games")
    parser.add_argument("--games", type=int, default=100, help="number of games to sample, spread over the file")
    parser.add_argument("--examples", type=int, default=0, help="also print this many positions where a theme was selected")
    parser.add_argument("--seed", type=int, default=0, help="random seed used to pick the examples")
    parser.add_argument("--first-ply", type=int, default=8, help="skip the first plies of every game (random openings)")
    args = parser.parse_args()
    with open(args.pgn, encoding="utf-8") as file:
        games = read_pgn(file.read())
    step = max(1, len(games) // args.games)
    fens: list[str] = []
    for game in games[::step][: args.games]:
        fens.extend(game_fens(game.moves, args.first_ply))
    model = MotifModel.load(args.model)
    scores = score_fens(model, fens)
    print(format_audit(selection_rates(scores, model.thresholds, model.themes)))
    if args.examples:
        print()
        print(format_examples(select_examples(scores, model.thresholds, model.themes, fens, args.examples, args.seed)))


if __name__ == "__main__":
    main()