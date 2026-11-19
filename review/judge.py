import math
from dataclasses import dataclass

from review.model import PlyAnalysis
from review.winprob import format_score, win_probability

CATEGORIES = ("best", "excellent", "good", "inaccuracy", "mistake", "blunder")
SKIPPED = "random"
SYMBOLS = {"best": "✓", "excellent": "", "good": "", "inaccuracy": "?!", "mistake": "?", "blunder": "??", SKIPPED: ""}
EXCELLENT_BELOW = 1.0
GOOD_BELOW = 5.0
INACCURACY_BELOW = 10.0
MISTAKE_BELOW = 20.0
KEY_MOMENT_LOSS = INACCURACY_BELOW
ACCURACY_DECAY = 0.04
DECISIVE_WIN = 80.0
SIMILAR_ACCURACY = 5.0
PHASES = (("Opening", 0, 20), ("Middlegame", 20, 60), ("Endgame", 60, 100_000))


@dataclass(frozen=True)
class ReviewedMove:
    ply: int
    number: int
    is_white: bool
    san: str
    uci: str
    fen: str
    best_san: str
    best_uci: str
    best_line: tuple[str, ...]
    white_cp_before: int
    white_cp_after: int
    win_before: float
    win_after: float
    loss: float
    category: str
    accuracy: float

    @property
    def label(self) -> str:
        dots = "." if self.is_white else "..."
        return f"{self.number}{dots} {self.san}"

    @property
    def symbol(self) -> str:
        return SYMBOLS[self.category]


@dataclass(frozen=True)
class SideSummary:
    moves: int
    accuracy: float | None
    average_loss: float | None
    counts: dict[str, int]
    phase_accuracy: dict[str, float | None]


def classify(loss: float, is_best: bool) -> str:
    if is_best:
        return "best"
    if loss < EXCELLENT_BELOW:
        return "excellent"
    if loss < GOOD_BELOW:
        return "good"
    if loss < INACCURACY_BELOW:
        return "inaccuracy"
    if loss < MISTAKE_BELOW:
        return "mistake"
    return "blunder"


def move_accuracy(loss: float) -> float:
    return 100.0 * math.exp(-ACCURACY_DECAY * max(loss, 0.0))


def _mover_win(white_centipawns: int, is_white: bool) -> float:
    white = win_probability(white_centipawns)
    return white if is_white else 100.0 - white


def review_moves(plies: list[PlyAnalysis], skip_plies: int = 0) -> list[ReviewedMove]:
    reviewed = []
    for item in plies:
        is_white = item.ply % 2 == 0
        before = _mover_win(item.white_cp_before, is_white)
        after = _mover_win(item.white_cp_after, is_white)
        skipped = item.ply < skip_plies
        loss = 0.0 if skipped else max(0.0, before - after)
        category = SKIPPED if skipped else classify(loss, item.uci == item.best_uci)
        reviewed.append(
            ReviewedMove(
                item.ply,
                item.ply // 2 + 1,
                is_white,
                item.san,
                item.uci,
                item.fen,
                item.best_san,
                item.best_uci,
                item.best_line,
                item.white_cp_before,
                item.white_cp_after,
                before,
                after,
                loss,
                category,
                0.0 if skipped else move_accuracy(loss),
            )
        )
    return reviewed


def summarize_side(moves: list[ReviewedMove], white: bool) -> SideSummary:
    own = [move for move in moves if move.is_white == white and move.category != SKIPPED]
    counts = {category: 0 for category in CATEGORIES}
    for move in own:
        counts[move.category] += 1
    phase_accuracy: dict[str, float | None] = {}
    for name, start, end in PHASES:
        subset = [move for move in own if start <= move.ply < end]
        phase_accuracy[name] = sum(m.accuracy for m in subset) / len(subset) if subset else None
    if not own:
        return SideSummary(0, None, None, counts, phase_accuracy)
    return SideSummary(
        len(own),
        sum(move.accuracy for move in own) / len(own),
        sum(move.loss for move in own) / len(own),
        counts,
        phase_accuracy,
    )


def key_moments(moves: list[ReviewedMove], count: int = 5, minimum_loss: float = KEY_MOMENT_LOSS) -> list[ReviewedMove]:
    candidates = [move for move in moves if move.loss >= minimum_loss]
    strongest = sorted(candidates, key=lambda move: -move.loss)[:count]
    return sorted(strongest, key=lambda move: move.ply)


def _white_win(mover_win: float, is_white: bool) -> float:
    return mover_win if is_white else 100.0 - mover_win


def _is_decisive(white_win: float) -> bool:
    return white_win >= DECISIVE_WIN or white_win <= 100.0 - DECISIVE_WIN


def decisive_move(moves: list[ReviewedMove]) -> ReviewedMove | None:
    for move in moves:
        if move.category == SKIPPED:
            continue
        before = _white_win(move.win_before, move.is_white)
        after = _white_win(move.win_after, move.is_white)
        if _is_decisive(before):
            continue
        if _is_decisive(after):
            return move
    return None


def _result_sentence(result: str, white: str, black: str) -> str:
    if result == "1-0":
        return f"{white} won the game."
    if result == "0-1":
        return f"{black} won the game."
    if result == "1/2-1/2":
        return "The game ended in a draw."
    return "The game has no recorded result."


def _worst(moves: list[ReviewedMove], white: bool) -> ReviewedMove | None:
    own = [move for move in moves if move.is_white == white and move.category != SKIPPED]
    worst = max(own, key=lambda move: move.loss, default=None)
    return worst if worst is not None and worst.loss >= INACCURACY_BELOW else None


def narrative(moves: list[ReviewedMove], result: str, white: str = "White", black: str = "Black") -> list[str]:
    lines = [_result_sentence(result, white, black)]
    white_summary = summarize_side(moves, True)
    black_summary = summarize_side(moves, False)
    if white_summary.accuracy is not None and black_summary.accuracy is not None:
        white_figure, black_figure = round(white_summary.accuracy), round(black_summary.accuracy)
        if abs(white_figure - black_figure) < SIMILAR_ACCURACY:
            lines.append(f"Both sides played with similar accuracy ({white_figure}% against {black_figure}%).")
        else:
            better, better_figure, worse_figure = (
                (white, white_figure, black_figure) if white_figure > black_figure else (black, black_figure, white_figure)
            )
            lines.append(f"{better} played more accurately ({better_figure}% against {worse_figure}%).")
    judged = [move for move in moves if move.category != SKIPPED]
    decisive = decisive_move(moves)
    if decisive is not None:
        white_win = _white_win(decisive.win_after, decisive.is_white)
        leader_is_white = white_win >= 50.0
        leader = white if leader_is_white else black
        mover = white if decisive.is_white else black
        score = format_score(decisive.white_cp_after)
        if decisive.is_white == leader_is_white:
            lines.append(f"{leader} reached a winning position with {decisive.label} (evaluation {score}).")
        else:
            quality = f"{decisive.category} " if decisive.category in ("mistake", "blunder") else ""
            lines.append(f"{leader} reached a winning position after {mover}'s {quality}{decisive.label} (evaluation {score}).")
    elif judged and _is_decisive(_white_win(judged[0].win_before, judged[0].is_white)):
        first = judged[0]
        leader = white if _white_win(first.win_before, first.is_white) >= 50.0 else black
        lines.append(
            f"{leader} was already winning before the first judged move "
            f"(evaluation {format_score(first.white_cp_before)}): the random opening decided the game."
        )
    else:
        lines.append("Neither side reached a clearly winning position.")
    for name, is_white in ((white, True), (black, False)):
        worst = _worst(moves, is_white)
        if worst is not None:
            lines.append(
                f"{name}'s biggest error was {worst.label} ({worst.category}): "
                f"winning chances fell from {worst.win_before:.0f}% to {worst.win_after:.0f}%."
            )
    return lines