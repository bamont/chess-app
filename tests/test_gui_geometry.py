from chess.pieces import Kind
from ui.gui import BOARD_PIXELS, SQUARE_SIZE, pixel_to_square, promotion_choice_at, square_to_pixel


def test_square_to_pixel_corners():
    assert square_to_pixel((0, 0)) == (0, 0)
    assert square_to_pixel((7, 7)) == (7 * SQUARE_SIZE, 7 * SQUARE_SIZE)
    assert square_to_pixel((7, 0)) == (0, 7 * SQUARE_SIZE)


def test_pixel_to_square_in_the_middle_of_a_square():
    half = SQUARE_SIZE // 2
    assert pixel_to_square(half, half) == (0, 0)
    assert pixel_to_square(4 * SQUARE_SIZE + half, 6 * SQUARE_SIZE + half) == (6, 4)


def test_round_trip_on_every_square():
    for row in range(8):
        for col in range(8):
            x, y = square_to_pixel((row, col))
            assert pixel_to_square(x, y) == (row, col)
            assert pixel_to_square(x + SQUARE_SIZE - 1, y + SQUARE_SIZE - 1) == (row, col)


def test_outside_the_board_returns_none():
    assert pixel_to_square(-1, 10) is None
    assert pixel_to_square(8 * SQUARE_SIZE, 10) is None
    assert pixel_to_square(10, 8 * SQUARE_SIZE) is None


def test_promotion_choice_at_each_cell():
    x0 = (BOARD_PIXELS - 4 * SQUARE_SIZE) // 2
    y = BOARD_PIXELS // 2
    kinds = [promotion_choice_at(x0 + i * SQUARE_SIZE + SQUARE_SIZE // 2, y) for i in range(4)]
    assert kinds == [Kind.QUEEN, Kind.ROOK, Kind.BISHOP, Kind.KNIGHT]


def test_promotion_choice_outside_the_picker_is_none():
    assert promotion_choice_at(5, 5) is None
    assert promotion_choice_at(BOARD_PIXELS // 2, 5) is None


def test_flipped_square_to_pixel_mirrors_the_board():
    assert square_to_pixel((0, 0), flipped=True) == (7 * SQUARE_SIZE, 7 * SQUARE_SIZE)
    assert square_to_pixel((7, 7), flipped=True) == (0, 0)
    assert square_to_pixel((6, 4), flipped=True) == (3 * SQUARE_SIZE, SQUARE_SIZE)


def test_flipped_round_trip_on_every_square():
    for row in range(8):
        for col in range(8):
            x, y = square_to_pixel((row, col), flipped=True)
            assert pixel_to_square(x, y, flipped=True) == (row, col)
            assert pixel_to_square(x + SQUARE_SIZE - 1, y + SQUARE_SIZE - 1, flipped=True) == (row, col)


def test_flipped_outside_the_board_returns_none():
    assert pixel_to_square(-1, 10, flipped=True) is None