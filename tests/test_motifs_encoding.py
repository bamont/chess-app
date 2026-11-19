import pytest

np = pytest.importorskip("numpy")

from motifs.encoding import PACKED_BYTES, PLANES, apply_uci, encode_position, pack_positions, parse_fen, unpack_positions  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def board(fen: str) -> list[list[str]]:
    return parse_fen(fen)[0]


def test_fen_is_parsed_into_rows_and_side_to_move():
    rows, white_to_move = parse_fen(START)
    assert white_to_move
    assert rows[0] == list("rnbqkbnr")
    assert rows[4] == ["."] * 8
    assert rows[7] == list("RNBQKBNR")
    assert parse_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1")[1] is False


def test_invalid_placements_are_rejected():
    with pytest.raises(ValueError):
        parse_fen("4k3/8/8/8/8/8/4K3 w - - 0 1")
    with pytest.raises(ValueError):
        parse_fen("4k4/8/8/8/8/8/8/4K3 w - - 0 1")


def test_a_plain_move_and_a_capture():
    rows = board(START)
    apply_uci(rows, "e2e4")
    assert rows[6][4] == "." and rows[4][4] == "P"
    rows = board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    apply_uci(rows, "e4d5")
    assert rows[3][3] == "P" and rows[4][4] == "."


def test_en_passant_removes_the_captured_pawn():
    rows = board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    apply_uci(rows, "e5d6")
    assert rows[2][3] == "P"
    assert rows[3][3] == "."
    assert rows[3][4] == "."


def test_castling_moves_the_rook():
    rows = board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    apply_uci(rows, "e1g1")
    assert rows[7][6] == "K" and rows[7][5] == "R" and rows[7][7] == "." and rows[7][4] == "."
    rows = board("r3k3/8/8/8/8/8/8/4K3 b q - 0 1")
    apply_uci(rows, "e8c8")
    assert rows[0][2] == "k" and rows[0][3] == "r" and rows[0][0] == "." and rows[0][4] == "."


def test_promotion_keeps_the_color():
    rows = board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1")
    apply_uci(rows, "a7a8q")
    assert rows[0][0] == "Q"
    rows = board("4k3/8/8/8/8/8/7p/4K3 b - - 0 1")
    apply_uci(rows, "h2h1n")
    assert rows[7][7] == "n"


def test_moving_from_an_empty_square_is_an_error():
    with pytest.raises(ValueError):
        apply_uci(board(START), "e4e5")


def test_the_start_position_is_encoded_from_the_side_to_move():
    planes = encode_position(board(START), True)
    assert planes.shape == (PLANES, 8, 8)
    assert int(planes.sum()) == 32
    assert planes[0, 6].tolist() == [1] * 8
    assert planes[6, 1].tolist() == [1] * 8
    assert planes[5, 7, 4] == 1
    assert planes[11, 0, 4] == 1


def test_black_to_move_is_mirrored_so_the_mover_is_always_at_the_bottom():
    rows = board(START)
    apply_uci(rows, "e2e4")
    planes = encode_position(rows, False)
    assert int(planes.sum()) == 32
    assert planes[0, 6].tolist() == [1] * 8
    assert planes[5, 7, 4] == 1
    assert planes[6, 3, 4] == 1
    assert planes[6, 1].tolist() == [1, 1, 1, 1, 0, 1, 1, 1]


def test_the_same_position_with_colors_swapped_gives_the_same_planes():
    white = encode_position(board("4k3/8/8/3p4/8/8/4P3/4K3 w - - 0 1"), True)
    black = encode_position(board("4k3/4p3/8/8/3P4/8/8/4K3 b - - 0 1"), False)
    assert np.array_equal(white, black)


def test_positions_survive_packing():
    rng = np.random.default_rng(0)
    planes = (rng.random((7, PLANES, 8, 8)) < 0.2).astype(np.uint8)
    packed = pack_positions(planes)
    assert packed.shape == (7, PACKED_BYTES)
    assert packed.dtype == np.uint8
    assert np.array_equal(unpack_positions(packed), planes)