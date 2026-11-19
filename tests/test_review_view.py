import pytest

from review.model import Candidate, PlyAnalysis
from ui.review_view import (
    ROW_HEIGHT,
    ReviewSession,
    arrow_shape,
    row_at,
    uci_squares,
    visible_start,
)


def ply(index, before, after, uci="e2e4", best="d2d4", candidates=(), san="e4", best_san="d4"):
    return PlyAnalysis(index, f"fen{index}", san, uci, best_san, best, ("d4", "d5", "c4"), before, after, tuple(candidates))


def candidate(san, uci, cp, line=()):
    return Candidate(san, uci, cp, tuple(line) or (san,))


def session(plies):
    return ReviewSession(plies, make_board=lambda fen: f"board:{fen}")


def game():
    return [
        ply(0, 30, 20, uci="d2d4", best="d2d4"),
        ply(1, 20, 20, uci="e7e5", best="d7d5"),
        ply(2, 20, -400, uci="h2h4", best="g1f3"),
        ply(3, -400, -420, uci="a7a6", best="a7a6"),
    ]


def test_squares_are_read_from_a_move():
    assert uci_squares("e2e4") == ((6, 4), (4, 4))
    assert uci_squares("a1h8") == ((7, 0), (0, 7))
    assert uci_squares("e7e8q") == ((1, 4), (0, 4))
    assert uci_squares("(none)") is None
    assert uci_squares("e2") is None


def test_the_visible_rows_follow_the_cursor():
    assert visible_start(0, 40, 14) == 0
    assert visible_start(20, 40, 14) == 13
    assert visible_start(39, 40, 14) == 26
    assert visible_start(3, 5, 14) == 0


def test_a_click_on_a_row_gives_its_index():
    assert row_at(100, 100, 5, 14, 40) == 5
    assert row_at(100 + ROW_HEIGHT * 3 + 1, 100, 5, 14, 40) == 8
    assert row_at(99, 100, 5, 14, 40) is None
    assert row_at(100 + ROW_HEIGHT * 14, 100, 5, 14, 40) is None
    assert row_at(100 + ROW_HEIGHT * 6, 100, 0, 14, 3) is None


def test_an_arrow_points_at_its_target():
    shaft_start, shaft_end, head = arrow_shape((0.0, 0.0), (100.0, 0.0), 80.0)
    assert shaft_start == (0.0, 0.0)
    assert shaft_end[0] == pytest.approx(64.0)
    assert shaft_end[1] == pytest.approx(0.0)
    assert head[0] == (100.0, 0.0)
    assert head[1][0] == pytest.approx(64.0)
    assert head[2][0] == pytest.approx(64.0)
    assert head[1][1] == pytest.approx(-head[2][1])


def test_an_arrow_without_length_is_not_drawn():
    assert arrow_shape((5.0, 5.0), (5.0, 5.0), 80.0) is None


def test_a_review_needs_a_game():
    with pytest.raises(ValueError):
        ReviewSession([])


def test_the_review_starts_at_the_first_move_and_stays_inside_the_game():
    review = session(game())
    assert review.cursor == 0
    assert len(review) == 4
    review.previous()
    assert review.cursor == 0
    review.last()
    assert review.cursor == 3
    review.next()
    assert review.cursor == 3
    review.first()
    review.next()
    assert review.cursor == 1
    review.go_to(99)
    assert review.cursor == 3
    review.go_to(-4)
    assert review.cursor == 0


def test_the_board_is_the_position_before_the_move_and_is_cached():
    made = []
    review = ReviewSession(game(), make_board=lambda fen: made.append(fen) or f"board:{fen}")
    assert review.board == "board:fen0"
    assert review.board == "board:fen0"
    review.next()
    assert review.board == "board:fen1"
    assert made == ["fen0", "fen1"]


def test_the_mistakes_can_be_jumped_to():
    review = session(game())
    review.next_mistake()
    assert review.cursor == 2
    review.next_mistake()
    assert review.cursor == 2
    review.previous_mistake()
    assert review.cursor == 2
    review.last()
    review.previous_mistake()
    assert review.cursor == 2
    review.first()
    review.previous_mistake()
    assert review.cursor == 0


def test_the_played_move_is_drawn_from_its_squares():
    review = session(game())
    assert review.played_squares == ((6, 3), (4, 3))
    assert review.played_is_best


def test_without_candidates_the_best_move_is_the_alternative():
    review = session(game())
    assert review.best_squares == []
    review.go_to(2)
    assert not review.played_is_best
    assert review.best_squares == [((7, 6), (5, 5))]


def test_candidates_replace_the_single_best_move_and_skip_the_played_one():
    plies = [
        ply(
            0,
            30,
            -50,
            uci="e2e4",
            best="d2d4",
            candidates=[candidate("d4", "d2d4", 30), candidate("e4", "e2e4", 25), candidate("Nf3", "g1f3", 20), candidate("c4", "c2c4", 10)],
        )
    ]
    review = session(plies)
    assert review.best_squares == [((6, 3), (4, 3)), ((7, 6), (5, 5))]


def test_the_rows_show_the_move_and_the_better_one():
    review = session(game())
    good = review.row(0)
    assert good.text == "1. e4"
    assert good.best == ""
    bad = review.row(2)
    assert bad.text.endswith("??")
    assert bad.best == "best d4"
    assert bad.category == "blunder"


def test_the_headline_names_the_quality_and_the_loss():
    review = session(game())
    assert review.headline() == "1. e4  best move"
    review.go_to(2)
    assert review.headline().startswith("2. e4  blunder ??")
    assert "(-" in review.headline()


def test_the_details_without_candidates_give_the_best_line():
    review = session(game())
    review.go_to(2)
    lines = review.detail_lines()
    assert lines[0] == "Evaluation +0.20 -> -4.00"
    assert lines[1] == "Best: d4"
    assert lines[2] == "d4 d5 c4"


def test_the_details_list_every_candidate_with_its_score_and_line():
    plies = [
        ply(
            0,
            30,
            25,
            uci="e2e4",
            best="d2d4",
            candidates=[candidate("d4", "d2d4", 30, ("d4", "d5", "c4")), candidate("e4", "e2e4", 25, ("e4", "e5"))],
        )
    ]
    lines = session(plies).detail_lines()
    assert lines[0] == "Evaluation +0.30 -> +0.25"
    assert lines[1] == "Engine's choices:"
    assert lines[2] == "1. d4  +0.30"
    assert lines[3] == "    d5 c4"
    assert lines[4] == "2. e4  +0.25  (played)"
    assert lines[5] == "    e5"
    assert len(lines) == 6


def test_a_played_move_outside_the_choices_is_added():
    plies = [ply(0, 30, -80, uci="h2h4", best="d2d4", san="h4", candidates=[candidate("d4", "d2d4", 30), candidate("Nf3", "g1f3", 28)])]
    lines = session(plies).detail_lines()
    assert lines[-1] == "Played: h4  -0.80"


def test_a_single_candidate_is_worded_in_the_singular():
    plies = [ply(0, 30, 30, candidates=[candidate("d4", "d2d4", 30)])]
    assert session(plies).detail_lines()[1] == "Engine's choice:"


def test_the_evaluations_are_those_of_the_positions_before_each_move():
    assert session(game()).evaluations() == [30, 20, 20, -400]