import argparse
import shutil
from pathlib import Path

KINDS = {"Pawn": "P", "Knight": "N", "Bishop": "B", "Rook": "R", "Queen": "Q", "King": "K"}
COLORS = {"White": "w", "Black": "b"}
DESTINATION = Path(__file__).resolve().parent.parent / "ui" / "assets" / "pieces"


def prepare(source: Path, outline: bool, size: int) -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    suffix = " Outline" if outline else ""
    for color_name, color_code in COLORS.items():
        for kind_name, kind_code in KINDS.items():
            file = source / color_name / f"{kind_name} {color_name}{suffix} {size}px.png"
            shutil.copyfile(file, DESTINATION / f"{color_code}{kind_code}.png")
    shutil.copyfile(source / "License.txt", DESTINATION / "LICENSE.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy and rename a downloaded piece set.")
    parser.add_argument("source", type=Path, help="folder of the unzipped piece pack")
    parser.add_argument("--outline", action="store_true", help="use the outlined variant")
    parser.add_argument("--size", type=int, default=288, help="source image size in pixels")
    args = parser.parse_args()
    prepare(args.source, args.outline, args.size)