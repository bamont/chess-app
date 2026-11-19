import pytest

from bot.evaluation import evaluate_positional
from bot.minimax_bot import MinimaxBot
from chess.board import Board
from chess.notation import move_to_uci

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
ENDGAME = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
PROMOTIONS = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
CHECKS = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POISONED_PAWN = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"


def test_without_quiescence_the_horizon_effect_wins_the_poisoned_pawn():
    bot = MinimaxBot(depth=1, seed=0, quiescence_limit=0)
    assert move_to_uci(bot.choose_move(Board.from_fen(POISONED_PAWN))) == "d1d5"


def test_quiescence_sees_the_recapture():
    bot = MinimaxBot(depth=1, seed=0, quiescence_limit=4)
    assert move_to_uci(bot.choose_move(Board.from_fen(POISONED_PAWN))) != "d1d5"


def test_quiescence_still_takes_a_free_queen():
    board = Board.from_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    bot = MinimaxBot(depth=1, seed=0, quiescence_limit=4)
    assert move_to_uci(bot.choose_move(board)) == "e4d5"


def test_quiescence_nodes_are_counted_separately():
    board = Board.from_fen(KIWIPETE)
    without = MinimaxBot(depth=1, seed=0, quiescence_limit=0)
    with_quiescence = MinimaxBot(depth=1, seed=0, quiescence_limit=3)
    without.search(board)
    with_quiescence.search(board)
    assert without.qnodes == 0
    assert with_quiescence.qnodes > 0


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError):
        MinimaxBot(depth=1, quiescence_limit=-1)


@pytest.mark.parametrize(
    "fen, depth",
    [
        (START, 1),
        (START, 2),
        (KIWIPETE, 1),
        (PROMOTIONS, 1),
        (CHECKS, 1),
        (ENDGAME, 2),
        (POISONED_PAWN, 2),
    ],
)
def test_pruning_matches_plain_negamax_with_quiescence(fen, depth):
    kwargs = {"depth": depth, "seed": 0, "evaluator": evaluate_positional, "quiescence_limit": 3}
    plain_score, plain_moves = MinimaxBot(pruning=False, **kwargs).search(Board.from_fen(fen))
    pruned_score, pruned_moves = MinimaxBot(pruning=True, **kwargs).search(Board.from_fen(fen))
    assert pruned_score == plain_score
    assert set(pruned_moves) == set(plain_moves)