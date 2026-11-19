import html
import math
import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from review.judge import (
    CATEGORIES,
    PHASES,
    SKIPPED,
    ReviewedMove,
    SideSummary,
    key_moments,
    narrative,
    review_moves,
    summarize_side,
)
from review.model import PlyAnalysis
from review.winprob import MATE_CENTIPAWNS, MATE_THRESHOLD, format_score, win_probability

SQUARE = 40
BOARD_SIZE = SQUARE * 8
PIECES = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟\ufe0e"}
SQUARE_PATTERN = re.compile(r"^[a-h][1-8]$")
GRAPH_WIDTH = 760
GRAPH_HEIGHT = 200
GRAPH_MARGIN = 28
WEAK_CATEGORIES = ("inaccuracy", "mistake", "blunder")
MOTIF_LIMIT = 3
TACTICAL_PLIES = 4
TACTICAL_MARKS = "x+#"
MATE_THEMES = ("mateIn1", "mateIn2", "mateIn3")
MATE_PATTERNS = ("backRankMate", "smotheredMate")
THEME_NAMES = {
    "xRayAttack": "x-ray attack",
    "backRankMate": "back-rank mate",
    "capturingDefender": "capturing the defender",
}
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z0-9])")

STYLE = """
:root {
  --bg: #f6f5f1; --fg: #1d2127; --muted: #66707c; --card: #ffffff; --border: #dcdad3;
  --light-sq: #f0d9b5; --dark-sq: #b58863;
  --graph-white: #fbfbf8; --graph-black: #4b5058; --graph-line: #1d2127; --graph-grid: #9aa0a8;
  --best: #2e8b57; --excellent: #4a9f78; --good: #8aa86b;
  --inaccuracy: #c9a227; --mistake: #e0801e; --blunder: #d64545;
  --arrow-best: #2e8b57; --arrow-played: #d64545;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #15171a; --fg: #e8e6e1; --muted: #9aa3ad; --card: #1f2226; --border: #333840;
    --light-sq: #c8b48f; --dark-sq: #8a6a4a;
    --graph-white: #d9d7d0; --graph-black: #2a2e34; --graph-line: #f2f0ea; --graph-grid: #6b727b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 32px 0 12px; }
.meta { color: var(--muted); margin: 0 0 16px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.scoreboard { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.side h3 { margin: 0; font-size: 1rem; overflow-wrap: anywhere; }
.accuracy { font-size: 2.2rem; font-weight: 700; line-height: 1.2; }
.phases { color: var(--muted); font-size: .85rem; margin: 4px 0 0; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 4px 8px; text-align: left; }
.counts th, .counts td { border-bottom: 1px solid var(--border); }
.counts td:not(:first-child), .counts th:not(:first-child) { text-align: right; }
.scroll { overflow-x: auto; }
.graph { width: 100%; height: auto; display: block; }
.graph .bg { fill: var(--graph-black); }
.graph .area { fill: var(--graph-white); }
.graph .curve { fill: none; stroke: var(--graph-line); stroke-width: 1.5; }
.graph .mid { stroke: var(--graph-grid); stroke-dasharray: 3 3; }
.graph .phase { stroke: var(--graph-grid); stroke-dasharray: 2 4; }
.graph text { fill: var(--graph-grid); font-size: 10px; }
.graph .dot { stroke: var(--graph-line); stroke-width: 1.2; }
.dot-mistake, .dot-blunder, .dot-inaccuracy { cursor: help; }
.dot-inaccuracy { fill: var(--inaccuracy); }
.dot-mistake { fill: var(--mistake); }
.dot-blunder { fill: var(--blunder); }
.dot-best, .dot-excellent, .dot-good { fill: var(--best); }
.moments { display: grid; gap: 12px; }
.moment { display: grid; grid-template-columns: 200px 1fr; gap: 16px; align-items: start; }
.moment h3 { margin: 0 0 4px; font-size: 1.05rem; }
.moment p { margin: 4px 0; }
.board { width: 100%; height: auto; display: block; border-radius: 4px; }
.sq-l { fill: var(--light-sq); }
.sq-d { fill: var(--dark-sq); }
.coord { font-size: 9px; fill: rgba(0, 0, 0, .55); font-family: system-ui, sans-serif; }
.piece { font-size: 34px; text-anchor: middle; dominant-baseline: central;
  font-family: "Segoe UI Symbol", "Noto Sans Symbols 2", "DejaVu Sans", "Apple Symbols", sans-serif; }
.piece-w { fill: #ffffff; stroke: #1b1b1b; stroke-width: 1px; paint-order: stroke; }
.piece-b { fill: #1b1b1b; stroke: #d8d8d8; stroke-width: .4px; paint-order: stroke; }
.arrow-best, .arrow-played { opacity: .85; }
.arrow-best { stroke: var(--arrow-best); }
.arrow-best-head { fill: var(--arrow-best); opacity: .85; }
.arrow-played { stroke: var(--arrow-played); }
.arrow-played-head { fill: var(--arrow-played); opacity: .85; }
.badge { display: inline-block; min-width: 1.6em; padding: 0 5px; margin-left: 4px; border-radius: 9px;
  color: #fff; font-size: .78rem; font-weight: 700; text-align: center; }
.cat-best { background: var(--best); }
.cat-inaccuracy { background: var(--inaccuracy); }
.cat-mistake { background: var(--mistake); }
.cat-blunder { background: var(--blunder); }
.moves th { color: var(--muted); font-weight: 400; width: 3em; }
.moves td.move { width: 50%; }
.moves tr:nth-child(even) { background: color-mix(in srgb, var(--border) 35%, transparent); }
.san { font-weight: 600; }
.san.muted { color: var(--muted); font-weight: 400; }
.tag { display: inline-block; padding: 0 8px; margin: 0 4px 4px 0; border: 1px solid var(--border);
  border-radius: 9px; font-size: .82rem; cursor: help; }
.score { color: var(--muted); font-size: .8rem; margin-left: 8px; }
.alt { color: var(--muted); font-size: .8rem; margin-left: 8px; }
ul.summary { margin: 0; padding-left: 20px; }
footer { color: var(--muted); font-size: .8rem; margin-top: 32px; }
@media (max-width: 560px) {
  .scoreboard, .moment { grid-template-columns: 1fr; }
  .moment .board { max-width: 320px; }
}
"""


class MotifPrediction(Protocol):
    theme: str
    score: float


class MotifPredictor(Protocol):
    def predict_fen(self, fen: str, limit: int | None = None) -> Sequence[MotifPrediction]: ...


def mate_in(move: ReviewedMove) -> int | None:
    score = move.white_cp_before if move.is_white else -move.white_cp_before
    if score < MATE_THRESHOLD:
        return None
    return max((MATE_CENTIPAWNS - score) // 10, 1)


def trusted_themes(move: ReviewedMove, predictions: Sequence[MotifPrediction], limit: int) -> list[tuple[str, float]]:
    mate_available = mate_in(move) is not None
    kept = []
    for item in predictions:
        if item.theme in MATE_THEMES or (item.theme in MATE_PATTERNS and not mate_available):
            continue
        kept.append((item.theme, item.score))
    return kept[:limit]


def has_tactical_line(line: Sequence[str]) -> bool:
    return any(any(mark in san for mark in TACTICAL_MARKS) for san in line[:TACTICAL_PLIES])


def theme_label(theme: str) -> str:
    return THEME_NAMES.get(theme) or CAMEL_BOUNDARY.sub(" ", theme).lower()


def _parse_placement(fen: str) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    for rank in fen.split()[0].split("/"):
        row: list[str | None] = []
        for char in rank:
            if char.isdigit():
                row.extend([None] * int(char))
            else:
                row.append(char)
        rows.append(row)
    return rows


def _center(square: str, flipped: bool) -> tuple[float, float]:
    file = ord(square[0]) - ord("a")
    rank = int(square[1])
    column = 7 - file if flipped else file
    row = rank - 1 if flipped else 8 - rank
    return (column + 0.5) * SQUARE, (row + 0.5) * SQUARE


def _arrow(start: str, end: str, css_class: str, flipped: bool) -> str:
    if not (SQUARE_PATTERN.match(start) and SQUARE_PATTERN.match(end)) or start == end:
        return ""
    x1, y1 = _center(start, flipped)
    x2, y2 = _center(end, flipped)
    length = math.hypot(x2 - x1, y2 - y1)
    unit_x, unit_y = (x2 - x1) / length, (y2 - y1) / length
    head = SQUARE * 0.34
    base_x, base_y = x2 - unit_x * head, y2 - unit_y * head
    half = head * 0.55
    side_x, side_y = -unit_y * half, unit_x * half
    points = (
        f"{x2:.1f},{y2:.1f} {base_x + side_x:.1f},{base_y + side_y:.1f} {base_x - side_x:.1f},{base_y - side_y:.1f}"
    )
    return (
        f'<line class="{css_class}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{base_x:.1f}" y2="{base_y:.1f}" '
        f'stroke-width="{SQUARE * 0.15:.1f}" stroke-linecap="round"/>'
        f'<polygon class="{css_class}-head" points="{points}"/>'
    )


def board_svg(fen: str, played_uci: str = "", best_uci: str = "", flipped: bool = False) -> str:
    placement = _parse_placement(fen)
    parts = [
        f'<svg class="board" viewBox="0 0 {BOARD_SIZE} {BOARD_SIZE}" role="img" '
        f'aria-label="Position before the move">'
    ]
    for row in range(8):
        for column in range(8):
            actual_row = 7 - row if flipped else row
            actual_column = 7 - column if flipped else column
            x, y = column * SQUARE, row * SQUARE
            shade = "sq-l" if (row + column) % 2 == 0 else "sq-d"
            parts.append(f'<rect class="{shade}" x="{x}" y="{y}" width="{SQUARE}" height="{SQUARE}"/>')
            if column == 0:
                parts.append(f'<text class="coord" x="{x + 2}" y="{y + 10}">{8 - actual_row}</text>')
            if row == 7:
                letter = chr(ord("a") + actual_column)
                parts.append(f'<text class="coord" x="{x + SQUARE - 2}" y="{y + SQUARE - 3}" text-anchor="end">{letter}</text>')
    for row in range(8):
        for column in range(8):
            actual_row = 7 - row if flipped else row
            actual_column = 7 - column if flipped else column
            piece = placement[actual_row][actual_column] if actual_row < len(placement) else None
            if piece is None:
                continue
            color = "piece-w" if piece.isupper() else "piece-b"
            x, y = (column + 0.5) * SQUARE, (row + 0.5) * SQUARE + 1
            parts.append(f'<text class="piece {color}" x="{x:.1f}" y="{y:.1f}">{PIECES[piece.lower()]}</text>')
    best, played = best_uci[:4], played_uci[:4]
    if best and best == played:
        parts.append(_arrow(best[:2], best[2:4], "arrow-best", flipped))
    else:
        if best:
            parts.append(_arrow(best[:2], best[2:4], "arrow-best", flipped))
        if played:
            parts.append(_arrow(played[:2], played[2:4], "arrow-played", flipped))
    parts.append("</svg>")
    return "".join(parts)


def eval_graph(moves: list[ReviewedMove], highlights: list[ReviewedMove]) -> str:
    series = [win_probability(moves[0].white_cp_before)] if moves else [50.0]
    series += [win_probability(move.white_cp_after) for move in moves]
    count = len(series) - 1
    width, height, margin = GRAPH_WIDTH, GRAPH_HEIGHT, GRAPH_MARGIN
    inner_width, inner_height = width - 2 * margin, height - 2 * margin

    def x(index: float) -> float:
        return margin + (inner_width * index / count if count else inner_width / 2)

    def y(win: float) -> float:
        return margin + inner_height * (1 - win / 100.0)

    points = " ".join(f"{x(index):.1f},{y(value):.1f}" for index, value in enumerate(series))
    area = f"{x(0):.1f},{y(0):.1f} {points} {x(count):.1f},{y(0):.1f}"
    parts = [
        f'<svg class="graph" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Evaluation graph: White\'s winning chances after each move">',
        f'<rect class="bg" x="{margin}" y="{margin}" width="{inner_width}" height="{inner_height}"/>',
        f'<polygon class="area" points="{area}"/>',
        f'<line class="mid" x1="{margin}" y1="{y(50):.1f}" x2="{width - margin}" y2="{y(50):.1f}"/>',
        f'<polyline class="curve" points="{points}"/>',
        f'<text x="{margin + 4}" y="{margin + 12}">White ahead</text>',
        f'<text x="{margin + 4}" y="{height - margin - 5}">Black ahead</text>',
    ]
    for name, start, _ in PHASES[1:]:
        if 0 < start < count:
            parts.append(
                f'<line class="phase" x1="{x(start):.1f}" y1="{margin}" x2="{x(start):.1f}" y2="{height - margin}"/>'
                f'<text x="{x(start) + 3:.1f}" y="{margin - 6}">{name}</text>'
            )
    for ply in range(20, count + 1, 20):
        parts.append(f'<text x="{x(ply):.1f}" y="{height - margin + 14}" text-anchor="middle">{ply // 2}</text>')
    for move in highlights:
        win = win_probability(move.white_cp_after)
        tip = f"{move.label} {move.symbol}: {move.category}, evaluation {format_score(move.white_cp_after)}"
        parts.append(
            f'<circle class="dot dot-{move.category}" cx="{x(move.ply + 1):.1f}" cy="{y(win):.1f}" r="5">'
            f"<title>{html.escape(tip)}</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _percent(value: float | None) -> str:
    return "–" if value is None else f"{value:.0f}%"


def _side_card(name: str, color: str, summary: SideSummary) -> str:
    phases = " · ".join(f"{phase} {_percent(value)}" for phase, value in summary.phase_accuracy.items())
    return (
        f'<div class="card side"><h3>{html.escape(name)} <span class="meta">({color})</span></h3>'
        f'<div class="accuracy">{_percent(summary.accuracy)}</div>'
        f'<div class="meta">accuracy over {summary.moves} moves</div>'
        f'<p class="phases">{html.escape(phases)}</p></div>'
    )


def _counts_table(white: SideSummary, black: SideSummary) -> str:
    rows = "".join(
        f"<tr><td>{category.capitalize()}</td><td>{white.counts[category]}</td><td>{black.counts[category]}</td></tr>"
        for category in CATEGORIES
    )
    return (
        '<table class="counts"><thead><tr><th>Move quality</th><th>White</th><th>Black</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


def _move_cell(move: ReviewedMove | None) -> str:
    if move is None:
        return "<td></td>"
    if move.category == SKIPPED:
        return (
            f'<td class="move"><span class="san muted">{html.escape(move.san)}</span>'
            f'<span class="score">{html.escape(format_score(move.white_cp_after))}</span>'
            '<span class="alt">random</span></td>'
        )
    badge = f'<span class="badge cat-{move.category}">{html.escape(move.symbol)}</span>' if move.symbol else ""
    alternative = (
        f'<span class="alt">best {html.escape(move.best_san)}</span>' if move.category in WEAK_CATEGORIES else ""
    )
    return (
        f'<td class="move"><span class="san">{html.escape(move.san)}</span>{badge}'
        f'<span class="score">{html.escape(format_score(move.white_cp_after))}</span>{alternative}</td>'
    )


def _moves_table(moves: list[ReviewedMove]) -> str:
    rows = []
    for index in range(0, len(moves), 2):
        black = moves[index + 1] if index + 1 < len(moves) else None
        rows.append(f'<tr><th scope="row">{index // 2 + 1}</th>{_move_cell(moves[index])}{_move_cell(black)}</tr>')
    return f'<table class="moves"><tbody>{"".join(rows)}</tbody></table>'


def _themes_line(themes: Sequence[tuple[str, float]] | None) -> str:
    if themes is None:
        return ""
    if not themes:
        return '<p class="meta">No tactical theme stood out for this position.</p>'
    tags = "".join(
        f'<span class="tag" title="model score {score * 100:.0f}%">{html.escape(theme_label(theme))}</span>'
        for theme, score in themes
    )
    return (
        f"<p>Possible tactical themes: {tags}</p>"
        '<p class="meta">Suggested by a small neural network trained on Lichess puzzles; treat it as a hint.</p>'
    )


def _mate_line(move: ReviewedMove) -> str:
    moves = mate_in(move)
    return f"<p>Stockfish saw a forced mate in {moves}.</p>" if moves else ""


def _moment_card(move: ReviewedMove, name: str, themes: Sequence[tuple[str, float]] | None = None) -> str:
    line = " ".join(move.best_line) if move.best_line else move.best_san
    return (
        '<article class="card moment">'
        f"{board_svg(move.fen, move.uci, move.best_uci, flipped=not move.is_white)}"
        f"<div><h3>{html.escape(move.label)} "
        f'<span class="badge cat-{move.category}">{html.escape(move.symbol)}</span></h3>'
        f"<p>{html.escape(name)} lost {move.loss:.0f} points of winning chances "
        f"({move.win_before:.0f}% → {move.win_after:.0f}%).</p>"
        f"<p>Evaluation: {html.escape(format_score(move.white_cp_before))} → "
        f"{html.escape(format_score(move.white_cp_after))}</p>"
        f"{_mate_line(move)}"
        f"<p>Best was <strong>{html.escape(move.best_san)}</strong>"
        f"{' · ' + html.escape(line) if len(move.best_line) > 1 else ''}</p>"
        f"{_themes_line(themes)}"
        f'<p class="meta">Position: <code>{html.escape(move.fen)}</code></p>'
        '<p class="meta">Green arrow: engine\'s choice. Red arrow: the move played.</p></div></article>'
    )


def render_report(
    headers: dict[str, str],
    result: str,
    moves: list[ReviewedMove],
    motifs: Mapping[int, Sequence[tuple[str, float]]] | None = None,
) -> str:
    white_name = headers.get("White", "White")
    black_name = headers.get("Black", "Black")
    white = summarize_side(moves, True)
    black = summarize_side(moves, False)
    moments = key_moments(moves)
    details = " · ".join(headers[key] for key in ("Event", "Site", "Date") if headers.get(key) and headers[key] != "?")
    title = f"{white_name} vs {black_name}"
    skipped = sum(1 for move in moves if move.category == SKIPPED)
    skipped_note = (
        f'<p class="meta">The first {skipped} plies were random opening moves and are not judged.</p>' if skipped else ""
    )
    summary_items = "".join(f"<li>{html.escape(line)}</li>" for line in narrative(moves, result, white_name, black_name))
    if moments:
        moment_cards = "".join(
            _moment_card(move, white_name if move.is_white else black_name, (motifs or {}).get(move.ply))
            for move in moments
        )
    else:
        moment_cards = '<p class="card">No mistakes or blunders were found in this game.</p>'
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Game review: {html.escape(title)}</title><style>{STYLE}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="meta">{html.escape(result)}{" · " + html.escape(details) if details else ""}</p>'
        f'{skipped_note}<ul class="summary card">{summary_items}</ul>'
        "<h2>Accuracy</h2>"
        f'<div class="scoreboard">{_side_card(white_name, "White", white)}{_side_card(black_name, "Black", black)}</div>'
        f'<div class="card" style="margin-top:12px">{_counts_table(white, black)}</div>'
        "<h2>Evaluation</h2>"
        f'<div class="card scroll">{eval_graph(moves, moments)}</div>'
        "<h2>Key moments</h2>"
        f'<div class="moments">{moment_cards}</div>'
        "<h2>Moves</h2>"
        f'<div class="card scroll">{_moves_table(moves)}</div>'
        "<footer>Evaluations come from a UCI engine; winning chances are a logistic function of the "
        "centipawn score. Move quality is judged by the loss of winning chances.</footer>"
        "</main></body></html>\n"
    )


def build_report(
    headers: dict[str, str],
    result: str,
    plies: list[PlyAnalysis],
    skip_plies: int = 0,
    motif_model: MotifPredictor | None = None,
) -> str:
    moves = review_moves(plies, skip_plies)
    motifs = None
    if motif_model is not None:
        motifs = {
            move.ply: trusted_themes(move, motif_model.predict_fen(move.fen), MOTIF_LIMIT)
            for move in key_moments(moves)
            if has_tactical_line(move.best_line)
        }
    return render_report(headers, result, moves, motifs)