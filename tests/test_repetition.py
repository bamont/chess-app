import pytest

from bot.minimax_bot import MinimaxBot
from chess.board import Board
from chess.notation import move_to_uci, parse_uci
from chess.rules import generate_legal_moves

QUEEN_ENDGAME = "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"


def play(board: Board, *ucis: str) -> None:
    for uci in ucis:
        board.make_move(parse_uci(uci, generate_legal_moves(board, board.turn)))


def shuffled_board(*extra: str) -> Board:
    board = Board.from_fen(QUEEN_ENDGAME)
    play(board, "a1a2", "e8d8", "a2a1", "d8e8", *extra)
    return board


def best_ucis(bot: MinimaxBot, board: Board) -> tuple[int, set[str]]:
    score, moves = bot.search(board)
    return score, {move_to_uci(move) for move in moves}


def test_without_history_the_move_is_acceptable():
    _, moves = best_ucis(MinimaxBot(depth=1, seed=0), Board.from_fen(QUEEN_ENDGAME))
    assert "a1a2" in moves


def test_bot_ahead_avoids_repeating_a_position():
    score, moves = best_ucis(MinimaxBot(depth=1, seed=0), shuffled_board())
    assert score > 0
    assert moves
    assert "a1a2" not in moves


def test_bot_behind_seeks_the_repetition():
    score, moves = best_ucis(MinimaxBot(depth=1, seed=0), shuffled_board("a1a2"))
    assert moves == {"e8d8"}
    assert score == 0


def test_repetition_detection_can_be_disabled():
    bot = MinimaxBot(depth=1, seed=0, repetition_aware=False)
    _, moves = best_ucis(bot, shuffled_board())
    assert "a1a2" in moves


@pytest.mark.parametrize("seed", range(10))
def test_choose_move_never_repeats_when_ahead(seed):
    move = MinimaxBot(depth=1, seed=seed).choose_move(shuffled_board())
    assert move_to_uci(move) != "a1a2"


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_pruning_matches_plain_negamax_with_repetitions(depth):
    plain = best_ucis(MinimaxBot(depth=depth, seed=0, pruning=False), shuffled_board())
    pruned = best_ucis(MinimaxBot(depth=depth, seed=0, pruning=True), shuffled_board())
    assert pruned == plain


def test_search_leaves_the_board_and_its_history_untouched():
    board = shuffled_board()
    before = board.position_key()
    history_length = len(board.history)
    MinimaxBot(depth=2, seed=0).search(board)
    assert board.position_key() == before
    assert len(board.history) == history_length