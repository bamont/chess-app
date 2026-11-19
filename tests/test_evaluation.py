from bot.evaluation import PIECE_VALUES, evaluate
from chess.board import Board
from chess.pieces import Kind


def test_start_position_is_balanced():
    assert evaluate(Board.starting_position()) == 0


def test_extra_queen_is_worth_900_for_the_side_to_move():
    assert evaluate(Board.from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")) == 900


def test_score_is_from_the_point_of_view_of_the_side_to_move():
    white = Board.from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    black = Board.from_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
    assert evaluate(black) == -evaluate(white)


def test_piece_values_are_ordered():
    values = PIECE_VALUES
    assert values[Kind.QUEEN] > values[Kind.ROOK] > values[Kind.BISHOP] >= values[Kind.KNIGHT] > values[Kind.PAWN]
    assert values[Kind.KING] == 0