import random

from chess.pieces import Color

COLOR_CHOICES = ("white", "black", "random")


def resolve_color(choice: str, rng: random.Random | None = None) -> Color:
    if choice == "white":
        return Color.WHITE
    if choice == "black":
        return Color.BLACK
    if choice == "random":
        return (rng or random).choice(list(Color))
    raise ValueError(f"Unknown color choice: {choice}")