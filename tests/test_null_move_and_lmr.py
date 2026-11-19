import pytest

from bot.evaluation import MATE_SCORE, evaluate_positional
from bot.minimax_bot import MinimaxBot
from chess.board import Board
from chess.notation import move_to_uci
from chess.pieces import Color

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
MIDDLEGAME = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
MATE_IN_TWO = "7k/8/8/8/8/8/R7/1R4K1 w - - 0 1"
FREE_QUEEN = "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"
POISONED_PAWN = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"
PAWN_ENDING = "8/8/p7/P7/8/8/1k6/7K w - - 0 1"

OPTIONS = [{"null_move": True}, {"lmr": True}, {"null_move": True, "lmr": True}]


def search_result(fen: str, depth: int, **options):
    bot = MinimaxBot(depth=depth, seed=0, evaluator=evaluate_positional, **options)
    score, moves = bot.search(Board.from_fen(fen))
    return score, {move_to_uci(move) for move in moves}


def test_null_move_flips_the_turn_and_clears_the_en_passant_square():
    board = Board.from_fen("4k3/8/8/8/4P3/8/8/4K3 b - e3 5 1")
    saved = board.make_null_move()
    assert board.turn == Color.WHITE
    assert board.en_passant is None
    assert board.halfmove_clock == 0
    assert board.zobrist == board.compute_zobrist()
    board.undo_null_move(saved)
    assert board.turn == Color.BLACK
    assert board.en_passant == (5, 4)
    assert board.halfmove_clock == 5
    assert board.zobrist == board.compute_zobrist()


def test_null_move_leaves_the_pieces_and_the_history_untouched():
    board = Board.from_fen(KIWIPETE)
    before = board.position_key()
    saved = board.make_null_move()
    board.undo_null_move(saved)
    assert board.position_key() == before
    assert board.history == []


@pytest.mark.parametrize("options", OPTIONS)
def test_mate_in_two_is_still_found(options):
    bot = MinimaxBot(depth=5, seed=0, **options)
    score, _ = bot.search(Board.from_fen(MATE_IN_TWO))
    assert score == MATE_SCORE - 3


@pytest.mark.parametrize("options", OPTIONS)
def test_free_queen_is_still_captured(options):
    bot = MinimaxBot(depth=4, seed=0, **options)
    assert move_to_uci(bot.choose_move(Board.from_fen(FREE_QUEEN))) == "e4d5"


@pytest.mark.parametrize("options", OPTIONS)
def test_poisoned_pawn_is_still_avoided(options):
    bot = MinimaxBot(depth=3, seed=0, quiescence_limit=4, **options)
    assert move_to_uci(bot.choose_move(Board.from_fen(POISONED_PAWN))) != "d1d5"


def test_null_move_is_never_used_in_a_pure_pawn_ending():
    assert search_result(PAWN_ENDING, 5, null_move=True) == search_result(PAWN_ENDING, 5)


@pytest.mark.parametrize("fen", [MIDDLEGAME, KIWIPETE])
def test_pruning_techniques_reduce_the_number_of_nodes(fen):
    plain = MinimaxBot(depth=4, seed=0, evaluator=evaluate_positional)
    reduced = MinimaxBot(depth=4, seed=0, evaluator=evaluate_positional, null_move=True, lmr=True)
    plain.search(Board.from_fen(fen))
    reduced.search(Board.from_fen(fen))
    assert reduced.nodes < plain.nodes


def test_limits_require_iterative_deepening():
    with pytest.raises(ValueError):
        MinimaxBot(depth=3, time_limit=1.0, iterative=False)
    with pytest.raises(ValueError):
        MinimaxBot(depth=3, node_limit=1000, iterative=False)