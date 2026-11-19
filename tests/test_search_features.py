import time

import pytest

from bot.evaluation import MATE_SCORE, evaluate_positional
from bot.minimax_bot import MinimaxBot, order_moves, score_from_tt, score_to_tt
from chess.board import Board
from chess.move import Move
from chess.notation import move_to_uci
from chess.rules import generate_legal_moves
from chess.utils import to_square

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
ENDGAME = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
MIDDLEGAME = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
POISONED_PAWN = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"
BACK_RANK_MATE = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"
MATE_IN_TWO = "7k/8/8/8/8/8/R7/1R4K1 w - - 0 1"
CAPTURES = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"

CASES = [
    (START, 3),
    (KIWIPETE, 2),
    (ENDGAME, 3),
    (POISONED_PAWN, 3),
    (BACK_RANK_MATE, 3),
    (MIDDLEGAME, 2),
]


def search_result(fen: str, depth: int, **options):
    bot = MinimaxBot(depth=depth, seed=0, evaluator=evaluate_positional, **options)
    score, moves = bot.search(Board.from_fen(fen))
    return score, {move_to_uci(move) for move in moves}


def test_mate_scores_are_relative_to_the_node_in_the_table():
    stored = score_to_tt(MATE_SCORE - 5, 2)
    assert stored == MATE_SCORE - 3
    assert score_from_tt(stored, 4) == MATE_SCORE - 7
    assert score_from_tt(score_to_tt(-(MATE_SCORE - 5), 2), 4) == -(MATE_SCORE - 7)


def test_ordinary_scores_are_stored_unchanged():
    assert score_to_tt(150, 3) == 150
    assert score_from_tt(-40, 3) == -40


@pytest.mark.parametrize("fen, depth", CASES)
def test_transposition_table_does_not_change_the_result(fen, depth):
    assert search_result(fen, depth, use_tt=True) == search_result(fen, depth, use_tt=False)


@pytest.mark.parametrize("fen, depth", CASES)
def test_iterative_deepening_does_not_change_the_result(fen, depth):
    assert search_result(fen, depth, iterative=True) == search_result(fen, depth, iterative=False)


@pytest.mark.parametrize("fen, depth", CASES)
def test_all_features_match_the_plain_negamax(fen, depth):
    assert search_result(fen, depth) == search_result(fen, depth, pruning=False)


@pytest.mark.parametrize("use_tt", [True, False])
def test_finds_the_shortest_mate_in_two(use_tt):
    bot = MinimaxBot(depth=4, seed=0, use_tt=use_tt)
    score, _ = bot.search(Board.from_fen(MATE_IN_TWO))
    assert score == MATE_SCORE - 3


def test_depth_reached_matches_the_requested_depth():
    bot = MinimaxBot(depth=3, seed=0)
    bot.search(Board.from_fen(START))
    assert bot.depth_reached == 3


def test_node_limit_stops_the_search_early_and_deterministically():
    board = Board.from_fen(KIWIPETE)
    before = board.position_key()
    legal = set(generate_legal_moves(board, board.turn))
    results = []
    for _ in range(2):
        bot = MinimaxBot(depth=8, seed=0, node_limit=400)
        score, moves = bot.search(board)
        assert 1 <= bot.depth_reached < 8
        assert moves
        assert all(move in legal for move in moves)
        results.append((score, set(moves), bot.depth_reached))
    assert results[0] == results[1]
    assert board.position_key() == before
    assert board.history == []


def test_time_limit_is_respected():
    bot = MinimaxBot(depth=64, seed=0, time_limit=0.3, evaluator=evaluate_positional)
    board = Board.from_fen(MIDDLEGAME)
    start = time.perf_counter()
    move = bot.choose_move(board)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
    assert bot.depth_reached >= 1
    assert move in generate_legal_moves(board, board.turn)
    assert board.history == []


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        MinimaxBot(depth=2, time_limit=0)
    with pytest.raises(ValueError):
        MinimaxBot(depth=2, node_limit=0)
    with pytest.raises(ValueError):
        MinimaxBot(depth=200)


def test_table_move_is_searched_first():
    board = Board.from_fen(CAPTURES)
    quiet = Move(to_square("e1"), to_square("e2"))
    moves = order_moves(board, generate_legal_moves(board, board.turn), tt_move=quiet)
    assert moves[0] == quiet


def test_killer_moves_come_right_after_captures():
    board = Board.from_fen(CAPTURES)
    quiet = Move(to_square("e1"), to_square("e2"))
    moves = order_moves(board, generate_legal_moves(board, board.turn), killers=(quiet,))
    assert move_to_uci(moves[0]) == "e4d5"
    assert moves[1] == quiet