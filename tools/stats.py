import math
from collections.abc import Iterable

Z_95 = 1.96
ELO_SCALE = math.log(10) / 400


def elo_difference(score: float) -> float | None:
    if score <= 0 or score >= 1:
        return None
    return -400 * math.log10(1 / score - 1)


def expected_score(rating_difference: float) -> float:
    return 1 / (1 + 10 ** (-rating_difference / 400))


def wilson_interval(score: float, games: int, z: float = Z_95) -> tuple[float, float]:
    denominator = 1 + z * z / games
    center = (score + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(score * (1 - score) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def elo_bound(score: float) -> str:
    if score <= 0:
        return "-inf"
    if score >= 1:
        return "+inf"
    return f"{round(elo_difference(score)):+d}"


def color_summary(entries: Iterable[tuple[bool, float]]) -> str:
    counts = {True: [0, 0, 0], False: [0, 0, 0]}
    for is_white, points in entries:
        slot = 0 if points == 1 else 2 if points == 0 else 1
        counts[is_white][slot] += 1
    parts = []
    for name, is_white in (("white", True), ("black", False)):
        wins, draws, losses = counts[is_white]
        games = wins + draws + losses
        score = (wins + 0.5 * draws) / games if games else 0.0
        parts.append(f"as {name}: +{wins} ={draws} -{losses} (score {score:.2f})")
    return ", ".join(parts)