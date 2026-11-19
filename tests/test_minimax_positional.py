import pytest

from bot.evaluation import evaluate_positional
from bot.minimax_bot import MinimaxBot
from chess.board import Board
from chess.notation import move_to_uci

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
ENDGAME = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"


@pytest.mark.parametrize(
    "fen, depth",
    [(START, 1), (START, 2), (KIWIPETE, 2), (ENDGAME, 2), (ENDGAME, 3)],
)
def test_pruning_matches_plain_negamax_with_positional_evaluation(fen, depth):
    plain = MinimaxBot(depth=depth, seed=0, pruning=False, evaluator=evaluate_positional)
    pruned = MinimaxBot(depth=depth, seed=0, pruning=True, evaluator=evaluate_positional)
    plain_score, plain_moves = plain.search(Board.from_fen(fen))
    pruned_score, pruned_moves = pruned.search(Board.from_fen(fen))
    assert pruned_score == plain_score
    assert set(pruned_moves) == set(plain_moves)


def test_positional_bot_still_finds_mate_in_one():
    board = Board.from_fen("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    bot = MinimaxBot(depth=2, seed=0, evaluator=evaluate_positional)
    assert move_to_uci(bot.choose_move(board)) == "a1a8"