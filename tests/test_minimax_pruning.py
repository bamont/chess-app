import pytest

from bot.minimax_bot import MinimaxBot, order_moves
from chess.board import Board
from chess.notation import move_to_uci
from chess.rules import generate_legal_moves

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
ENDGAME = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
PROMOTIONS = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
CHECKS = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POISONED_PAWN = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"
BACK_RANK_MATE = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"

CASES = [(fen, depth) for fen in (START, KIWIPETE, ENDGAME, PROMOTIONS, CHECKS) for depth in (1, 2)]
CASES += [(fen, depth) for fen in (ENDGAME, POISONED_PAWN, BACK_RANK_MATE) for depth in (1, 2, 3)]


@pytest.mark.parametrize("fen, depth", CASES)
def test_pruning_matches_plain_negamax(fen, depth):
    plain_score, plain_moves = MinimaxBot(depth=depth, seed=0, pruning=False).search(Board.from_fen(fen))
    pruned_score, pruned_moves = MinimaxBot(depth=depth, seed=0, pruning=True).search(Board.from_fen(fen))
    assert pruned_score == plain_score
    assert set(pruned_moves) == set(plain_moves)


def test_pruning_visits_fewer_nodes():
    board = Board.starting_position()
    plain = MinimaxBot(depth=3, seed=0, pruning=False)
    pruned = MinimaxBot(depth=3, seed=0, pruning=True)
    plain.search(board)
    pruned.search(board)
    assert pruned.nodes < plain.nodes


def test_search_leaves_the_board_untouched():
    board = Board.from_fen(KIWIPETE)
    before = board.position_key()
    MinimaxBot(depth=3, seed=0).search(board)
    assert board.position_key() == before
    assert board.history == []


def test_last_score_is_recorded():
    bot = MinimaxBot(depth=2, seed=0)
    assert bot.last_score is None
    bot.choose_move(Board.from_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"))
    assert bot.last_score == 100


def test_captures_are_searched_first():
    board = Board.from_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    moves = order_moves(board, generate_legal_moves(board, board.turn))
    assert move_to_uci(moves[0]) == "e4d5"


def test_most_valuable_victim_comes_first():
    board = Board.from_fen("4k3/8/8/3p1q2/4P3/8/8/4K3 w - - 0 1")
    moves = order_moves(board, generate_legal_moves(board, board.turn))
    assert [move_to_uci(move) for move in moves[:2]] == ["e4f5", "e4d5"]


def test_queen_promotion_comes_first():
    board = Board.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = order_moves(board, generate_legal_moves(board, board.turn))
    assert move_to_uci(moves[0]) == "a7a8q"