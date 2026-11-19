from collections.abc import Sequence

from ui.eval_bar import white_share


def curve_values(white_scores: Sequence[int]) -> list[float]:
    return [white_share(score) for score in white_scores]


def curve_points(
    values: Sequence[float], left: float, top: float, width: float, height: float, flipped: bool
) -> list[tuple[float, float]]:
    count = len(values)
    step = width / (count - 1) if count > 1 else 0.0
    points = []
    for index, value in enumerate(values):
        offset = value if flipped else 1.0 - value
        points.append((left + index * step, top + offset * height))
    return points


def curve_x(index: int, left: float, width: float, count: int) -> float:
    if count <= 1:
        return left
    return left + width * index / (count - 1)


def curve_index_at(x: float, left: float, width: float, count: int) -> int | None:
    if count == 0 or x < left or x > left + width:
        return None
    if count == 1:
        return 0
    return max(0, min(count - 1, round((x - left) / width * (count - 1))))