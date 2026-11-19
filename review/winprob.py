import math

from tools.uci_engine import MATE_CENTIPAWNS

WIN_SLOPE = 0.00368
MATE_THRESHOLD = MATE_CENTIPAWNS - 1000


def win_probability(centipawns: int) -> float:
    return 100.0 / (1.0 + math.exp(-WIN_SLOPE * centipawns))


def format_score(white_centipawns: int) -> str:
    if white_centipawns >= MATE_CENTIPAWNS:
        return "1-0"
    if white_centipawns <= -MATE_CENTIPAWNS:
        return "0-1"
    if abs(white_centipawns) >= MATE_THRESHOLD:
        moves = (MATE_CENTIPAWNS - abs(white_centipawns)) // 10
        return f"M{moves}" if white_centipawns > 0 else f"-M{moves}"
    return f"{white_centipawns / 100:+.2f}"