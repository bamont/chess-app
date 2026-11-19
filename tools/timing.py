import time
from collections.abc import Callable


def best_of(function: Callable[[], object], repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def _calibration_work() -> None:
    total = 0
    for value in range(500_000):
        total += value * value % 7


def calibrate(repeat: int = 3) -> float:
    return best_of(_calibration_work, repeat)