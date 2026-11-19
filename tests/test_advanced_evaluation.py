import random

import pytest

import bot.advanced_evaluation as advanced
from bot.advanced_evaluation import evaluate_advanced, evaluate_advanced_without_shield
from bot.evaluation import evaluate_positional
from chess.board import Board
from chess.pieces import Piece
from chess.rules import generate_legal_moves

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1",
    "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1",
    "4k3/8/8/4P3/8/8/8/4K3 w - - 0 1",
    "4k3/3p4/8/4P3/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/4P3/4P3/4K3 w - - 0 1",
    "7k/8/8/8/8/8/R7/K7 w - - 0 1",
    "8/8/8/4k3/8/8/R7/K7 b - - 0 1",
    "8/8/8/8/8/8/8/8 w - - 0 1",
    "8/8/8/8/8/8/8/4K3 w - - 0 1",
]


def mirror(board: Board) -> Board:
    mirrored = Board()
    for row in range(8):
        for col in range(8):
            piece = board.grid[row][col]
            if piece is not None:
                mirrored.grid[7 - row][col] = Piece(piece.color.opponent(), piece.kind)
    mirrored.turn = board.turn.opponent()
    return mirrored


def evaluation_without(monkeypatch, fen: str, **constants) -> int:
    for name, value in constants.items():
        monkeypatch.setattr(advanced, name, value)
    return evaluate_advanced(Board.from_fen(fen))


@pytest.mark.parametrize("fen", POSITIONS)
def test_a_position_and_its_mirror_have_the_same_evaluation(fen):
    board = Board.from_fen(fen)
    assert evaluate_advanced(board) == evaluate_advanced(mirror(board))


@pytest.mark.parametrize("seed", range(5))
def test_mirror_symmetry_along_random_games(seed):
    rng = random.Random(seed)
    board = Board.starting_position()
    for _ in range(100):
        assert evaluate_advanced(board) == evaluate_advanced(mirror(board))
        moves = generate_legal_moves(board, board.turn)
        if not moves:
            break
        board.make_move(rng.choice(moves))


def test_start_position_is_balanced():
    assert evaluate_advanced(Board.starting_position()) == 0


def test_score_flips_with_the_side_to_move():
    white = Board.from_fen("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1")
    black = Board.from_fen("4k3/8/8/4P3/8/8/8/4K3 b - - 0 1")
    assert evaluate_advanced(black) == -evaluate_advanced(white)


def test_white_passed_pawn_bonus(monkeypatch):
    fen = "4k3/8/8/4P3/8/8/8/4K3 w - - 0 1"
    full = evaluate_advanced(Board.from_fen(fen))
    expected = advanced.PASSED_PAWN_EG[3]
    baseline = evaluation_without(monkeypatch, fen, PASSED_PAWN_MG=(0,) * 6, PASSED_PAWN_EG=(0,) * 6)
    assert full - baseline == expected


def test_black_passed_pawn_bonus(monkeypatch):
    fen = "4k3/8/8/8/4p3/8/8/4K3 b - - 0 1"
    full = evaluate_advanced(Board.from_fen(fen))
    expected = advanced.PASSED_PAWN_EG[3]
    baseline = evaluation_without(monkeypatch, fen, PASSED_PAWN_MG=(0,) * 6, PASSED_PAWN_EG=(0,) * 6)
    assert full - baseline == expected


def test_blocked_pawns_get_no_passed_bonus(monkeypatch):
    fen = "4k3/3p4/8/4P3/8/8/8/4K3 w - - 0 1"
    full = evaluate_advanced(Board.from_fen(fen))
    baseline = evaluation_without(monkeypatch, fen, PASSED_PAWN_MG=(0,) * 6, PASSED_PAWN_EG=(0,) * 6)
    assert full == baseline


def test_doubled_and_isolated_pawns_are_penalized(monkeypatch):
    fen = "4k3/8/8/8/8/4P3/4P3/4K3 w - - 0 1"
    full = evaluate_advanced(Board.from_fen(fen))
    expected = advanced.DOUBLED_PAWN_EG + 2 * advanced.ISOLATED_PAWN_EG
    baseline = evaluation_without(
        monkeypatch,
        fen,
        DOUBLED_PAWN_MG=0,
        DOUBLED_PAWN_EG=0,
        ISOLATED_PAWN_MG=0,
        ISOLATED_PAWN_EG=0,
    )
    assert baseline - full == expected


def test_bishop_pair_bonus(monkeypatch):
    fen = "4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1"
    full = evaluate_advanced(Board.from_fen(fen))
    phase = 2
    expected = (advanced.BISHOP_PAIR_MG * phase + advanced.BISHOP_PAIR_EG * (24 - phase)) // 24
    baseline = evaluation_without(monkeypatch, fen, BISHOP_PAIR_MG=0, BISHOP_PAIR_EG=0)
    assert full - baseline == expected


def test_mop_up_rewards_the_cornered_king(monkeypatch):
    fen = "7k/8/8/8/8/8/R7/K7 w - - 0 1"
    full = evaluate_advanced(Board.from_fen(fen))
    expected = advanced.MOP_UP_CENTER_WEIGHT * 6 + advanced.MOP_UP_PROXIMITY_WEIGHT * 0
    baseline = evaluation_without(monkeypatch, fen, MOP_UP_CENTER_WEIGHT=0, MOP_UP_PROXIMITY_WEIGHT=0)
    assert full - baseline == expected


def test_mop_up_is_smaller_when_the_king_is_in_the_center(monkeypatch):
    fen = "8/8/8/4k3/8/8/R7/K7 w - - 0 1"
    full = evaluate_advanced(Board.from_fen(fen))
    expected = advanced.MOP_UP_CENTER_WEIGHT * 0 + advanced.MOP_UP_PROXIMITY_WEIGHT * 6
    baseline = evaluation_without(monkeypatch, fen, MOP_UP_CENTER_WEIGHT=0, MOP_UP_PROXIMITY_WEIGHT=0)
    assert full - baseline == expected


def test_no_mop_up_when_material_is_balanced(monkeypatch):
    fen = "r3k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    full = evaluate_advanced(Board.from_fen(fen))
    baseline = evaluation_without(monkeypatch, fen, MOP_UP_CENTER_WEIGHT=0, MOP_UP_PROXIMITY_WEIGHT=0)
    assert full == baseline


def test_a_minor_piece_alone_is_nearly_a_draw():
    board = Board.from_fen("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1")
    assert abs(evaluate_advanced(board)) * 2 < abs(evaluate_positional(board))


def test_the_same_lead_with_a_pawn_is_still_a_win():
    assert evaluate_advanced(Board.from_fen("4k3/8/8/8/8/8/4P3/2B1K3 w - - 0 1")) > 350


SHIELD_POSITIONS = [
    "r1b1k3/pp3pp1/2p5/2np4/3Pp1q1/P1N3P1/1PP1P2K/R1BQ1R2 w q - 1 18",
    "7r/4k1p1/pqp1bp1p/4p2B/1b6/6P1/PQP1PP1P/K2R3R w - - 3 24",
    "r1b2knr/1pp2pp1/p1nb4/3N3P/7q/5PN1/PPPP2BP/R1BQK2R w KQ - 3 11",
    "3q2k1/pppppppp/8/8/8/8/PPPPPPPP/3Q2K1 w - - 0 1",
    "3q2k1/pppppppp/8/8/8/8/PPPPP1PP/3Q2K1 w - - 0 1",
]


def shield_delta(board: Board) -> int:
    return evaluate_advanced(board) - evaluate_advanced_without_shield(board)


@pytest.mark.parametrize("fen", POSITIONS + SHIELD_POSITIONS)
def test_the_evaluation_without_shield_is_mirror_symmetric(fen):
    board = Board.from_fen(fen)
    assert evaluate_advanced_without_shield(board) == evaluate_advanced_without_shield(mirror(board))


@pytest.mark.parametrize("fen", SHIELD_POSITIONS)
def test_the_shield_term_is_mirror_symmetric(fen):
    board = Board.from_fen(fen)
    assert shield_delta(board) == shield_delta(mirror(board))


def test_an_intact_shield_costs_nothing():
    board = Board.from_fen("3q2k1/pppppppp/8/8/8/8/PPPPPPPP/3Q2K1 w - - 0 1")
    assert shield_delta(board) == 0


def test_a_missing_shield_pawn_is_penalized():
    board = Board.from_fen("3q2k1/pppppppp/8/8/8/8/PPPPP1PP/3Q2K1 w - - 0 1")
    expected = -(advanced.SHIELD_MISSING * 8 // 24)
    assert abs(shield_delta(board) - expected) <= 1


def test_an_advanced_shield_pawn_is_penalized_less():
    board = Board.from_fen("3q2k1/pppppppp/8/8/8/5P2/PPPPP1PP/3Q2K1 w - - 0 1")
    expected = -(advanced.SHIELD_ADVANCED * 8 // 24)
    assert abs(shield_delta(board) - expected) <= 1


def test_the_king_file_without_a_pawn_costs_more():
    board = Board.from_fen("3q2k1/pppppppp/8/8/8/8/PPPPPP1P/3Q2K1 w - - 0 1")
    expected = -((advanced.SHIELD_MISSING + advanced.SHIELD_OPEN_FILE) * 8 // 24)
    assert abs(shield_delta(board) - expected) <= 1


def test_a_king_in_the_center_has_no_shield_penalty():
    board = Board.from_fen("3qk3/pppppppp/8/8/8/8/PPPPP1PP/3QK3 w - - 0 1")
    assert shield_delta(board) == 0