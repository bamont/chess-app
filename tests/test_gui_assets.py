import pytest

from chess.pieces import Color, Kind
from ui.gui import find_piece_file


@pytest.mark.parametrize("color", list(Color))
@pytest.mark.parametrize("kind", list(Kind))
def test_piece_image_exists(color, kind):
    assert find_piece_file(f"{color.value}{kind.value}").exists()