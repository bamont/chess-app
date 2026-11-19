from review.winprob import MATE_CENTIPAWNS, MATE_THRESHOLD, win_probability


def white_share(white_cp: int) -> float:
    if white_cp >= MATE_THRESHOLD:
        return 1.0
    if white_cp <= -MATE_THRESHOLD:
        return 0.0
    return win_probability(white_cp) / 100.0


def white_region(white_cp: int, height: int, flipped: bool) -> tuple[int, int]:
    size = round(white_share(white_cp) * height)
    top = 0 if flipped else height - size
    return top, size


def label_at_top(white_cp: int, flipped: bool) -> bool:
    return (white_cp >= 0) == flipped


def bar_label(white_cp: int) -> str:
    if abs(white_cp) >= MATE_THRESHOLD:
        moves = (MATE_CENTIPAWNS - abs(white_cp)) // 10
        return f"M{moves}" if moves > 0 else "#"
    return f"{abs(white_cp) / 100:.1f}"