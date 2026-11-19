import pytest

from bot.evaluation import MATE_SCORE, evaluate_positional
from bot.minimax_bot import MinimaxBot
from chess.board import Board
from chess.notation import move_to_uci
from chess.rules import generate_legal_moves

SACRIFICE_MATE = "2r4k/1pp2Qpp/p1b5/P1Pp4/3p3q/4P2P/6P1/B4RK1 b - - 0 29"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
MIDDLEGAME = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
MATE_IN_TWO = "7k/8/8/8/8/8/R7/1R4K1 w - - 0 1"


def ucis(moves) -> set[str]:
    return {move_to_uci(move) for move in moves}


def test_a_mate_found_inside_the_quiescence_search_is_seen():
    bot = MinimaxBot(depth=3, seed=0, evaluator=evaluate_positional, quiescence_limit=4)
    _, moves = bot.search(Board.from_fen(SACRIFICE_MATE))
    assert "c6b5" not in ucis(moves)


@pytest.mark.parametrize(
    "fen, depth",
    [(SACRIFICE_MATE, 2), (KIWIPETE, 1), (MIDDLEGAME, 1), (MATE_IN_TWO, 3)],
)
def test_pruned_and_plain_searches_agree_with_check_aware_quiescence(fen, depth):
    options = {"depth": depth, "seed": 0, "evaluator": evaluate_positional, "quiescence_limit": 3}
    plain_score, plain_moves = MinimaxBot(pruning=False, **options).search(Board.from_fen(fen))
    pruned_score, pruned_moves = MinimaxBot(pruning=True, **options).search(Board.from_fen(fen))
    assert pruned_score == plain_score
    assert ucis(pruned_moves) == ucis(plain_moves)


@pytest.mark.parametrize("fen", [KIWIPETE, MIDDLEGAME, SACRIFICE_MATE])
def test_check_extension_returns_a_legal_move_and_keeps_the_board(fen):
    board = Board.from_fen(fen)
    before = board.position_key()
    bot = MinimaxBot(depth=3, seed=0, check_extension=True, quiescence_limit=3)
    move = bot.choose_move(board)
    assert move in generate_legal_moves(board, board.turn)
    assert board.position_key() == before
    assert board.history == []


def test_check_extension_searches_more_nodes():
    with_extension = MinimaxBot(depth=3, seed=0, check_extension=True)
    without = MinimaxBot(depth=3, seed=0)
    with_extension.search(Board.from_fen(SACRIFICE_MATE))
    without.search(Board.from_fen(SACRIFICE_MATE))
    assert with_extension.nodes > without.nodes


def test_check_extension_still_finds_the_shortest_mate():
    bot = MinimaxBot(depth=4, seed=0, check_extension=True)
    score, _ = bot.search(Board.from_fen(MATE_IN_TWO))
    assert score == MATE_SCORE - 3