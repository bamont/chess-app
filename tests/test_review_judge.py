import pytest

from review.judge import (
    ReviewedMove,
    classify,
    decisive_move,
    key_moments,
    move_accuracy,
    narrative,
    review_moves,
    summarize_side,
)
from review.model import PlyAnalysis
from review.winprob import format_score, win_probability


def ply(index: int, before: int, after: int, uci: str = "e2e4", best: str = "d2d4") -> PlyAnalysis:
    return PlyAnalysis(index, "fen", "e4", uci, "d4", best, ("d4",), before, after)


def reviewed(index: int, loss: float, category: str = "blunder", accuracy: float = 50.0) -> ReviewedMove:
    return ReviewedMove(
        index, index // 2 + 1, index % 2 == 0, "e4", "e2e4", "fen", "d4", "d2d4", ("d4",), 0, 0, 50.0, 50.0 - loss, loss, category, accuracy
    )


def test_win_probability_is_centered_and_symmetric():
    assert win_probability(0) == 50.0
    assert win_probability(150) + win_probability(-150) == pytest.approx(100.0)


def test_win_probability_grows_with_the_score():
    assert win_probability(-300) < win_probability(0) < win_probability(100) < win_probability(900)
    assert win_probability(9000) > 99.9


def test_scores_are_formatted_from_whites_point_of_view():
    assert format_score(35) == "+0.35"
    assert format_score(-120) == "-1.20"
    assert format_score(0) == "+0.00"
    assert format_score(9970) == "M3"
    assert format_score(-9980) == "-M2"
    assert format_score(10000) == "1-0"
    assert format_score(-10000) == "0-1"


@pytest.mark.parametrize(
    "loss, expected",
    [(0.0, "excellent"), (0.99, "excellent"), (1.0, "good"), (4.9, "good"), (5.0, "inaccuracy"), (9.9, "inaccuracy"), (10.0, "mistake"), (19.9, "mistake"), (20.0, "blunder"), (80.0, "blunder")],
)
def test_classification_thresholds(loss, expected):
    assert classify(loss, False) == expected


def test_the_engines_move_is_always_best():
    assert classify(0.0, True) == "best"
    assert classify(3.0, True) == "best"


def test_accuracy_falls_with_the_loss():
    assert move_accuracy(0.0) == 100.0
    assert move_accuracy(10.0) > move_accuracy(20.0) > move_accuracy(50.0) > 0.0
    assert move_accuracy(-5.0) == 100.0


def test_loss_is_measured_from_the_movers_side():
    moves = review_moves([ply(0, 0, -300), ply(1, -300, -300)])
    assert moves[0].category == "blunder"
    assert moves[0].loss == pytest.approx(win_probability(0) - win_probability(-300))
    assert moves[0].win_before == pytest.approx(50.0)
    assert moves[1].loss == 0.0
    assert moves[1].category == "excellent"
    assert moves[1].win_before == pytest.approx(win_probability(300))


def test_black_losses_use_blacks_winning_chances():
    moves = review_moves([ply(1, 0, 300)])
    assert moves[0].category == "blunder"
    assert moves[0].loss == pytest.approx(win_probability(0) - win_probability(-300))


def test_an_improving_move_has_no_loss():
    moves = review_moves([ply(0, 0, 200)])
    assert moves[0].loss == 0.0


def test_move_labels_follow_the_pgn_convention():
    moves = review_moves([ply(0, 0, 0), ply(1, 0, 0), ply(2, 0, 0), ply(3, 0, 0)])
    assert [move.label for move in moves] == ["1. e4", "1... e4", "2. e4", "2... e4"]


def test_symbols_follow_the_category():
    moves = review_moves([ply(0, 0, -300), ply(1, -300, -300, uci="d2d4")])
    assert moves[0].symbol == "??"
    assert moves[1].symbol == "✓"


def test_side_summary_counts_categories_and_averages_accuracy():
    moves = review_moves([ply(0, 0, -300), ply(1, -300, -300), ply(2, -300, -300, uci="d2d4"), ply(3, -300, -300)])
    white = summarize_side(moves, True)
    assert white.moves == 2
    assert white.counts["blunder"] == 1
    assert white.counts["best"] == 1
    assert white.accuracy == pytest.approx((moves[0].accuracy + moves[2].accuracy) / 2)
    assert white.average_loss == pytest.approx(moves[0].loss / 2)
    assert white.phase_accuracy["Opening"] == pytest.approx(white.accuracy)
    assert white.phase_accuracy["Middlegame"] is None


def test_side_summary_without_moves():
    summary = summarize_side([], True)
    assert summary.moves == 0
    assert summary.accuracy is None
    assert summary.average_loss is None


def test_key_moments_keep_the_largest_losses_in_game_order():
    moves = [reviewed(i, loss) for i, loss in enumerate([5, 25, 12, 30, 22, 21, 40])]
    selected = key_moments(moves, count=3)
    assert [move.ply for move in selected] == [1, 3, 6]


def test_key_moments_ignore_small_losses():
    assert key_moments([reviewed(0, 9.0), reviewed(1, 3.0)]) == []


def test_key_moments_include_mistakes():
    assert [move.ply for move in key_moments([reviewed(0, 10.0), reviewed(1, 3.0)])] == [0]


def test_decisive_move_is_the_first_winning_position():
    moves = review_moves([ply(0, 0, 0), ply(1, 0, 0), ply(2, 0, 500), ply(3, 500, 600)])
    assert decisive_move(moves).ply == 2


def test_decisive_move_for_black():
    moves = review_moves([ply(0, 0, 0), ply(1, 0, -500)])
    assert decisive_move(moves).ply == 1


def test_no_decisive_move_in_a_balanced_game():
    assert decisive_move(review_moves([ply(0, 0, 20), ply(1, 20, 10)])) is None


def test_narrative_names_the_winner_and_the_biggest_error():
    moves = review_moves([ply(0, 0, -300), ply(1, -300, -300)])
    lines = narrative(moves, "0-1", "Alice", "Bob")
    assert lines[0] == "Bob won the game."
    assert any(line.startswith("Bob played more accurately") for line in lines)
    assert any("Alice's biggest error was 1. e4 (blunder)" in line for line in lines)
    assert any("Neither side reached a clearly winning position." in line for line in lines)


def test_narrative_blames_the_opponents_move_that_gave_the_win():
    moves = review_moves([ply(0, 0, 0), ply(1, 0, 500, uci="d7d5")])
    lines = narrative(moves, "1-0")
    assert any("White reached a winning position after Black's blunder 1... e4 (evaluation +5.00)" in line for line in lines)


def test_narrative_credits_the_move_that_gave_the_win():
    lines = narrative(review_moves([ply(0, 0, 500)]), "1-0")
    assert any("White reached a winning position with 1. e4 (evaluation +5.00)" in line for line in lines)


def test_accuracy_verdict_follows_the_displayed_figures():
    far = [reviewed(0, 0.0, "good", 94.6), reviewed(1, 0.0, "good", 90.2)]
    near = [reviewed(0, 0.0, "good", 94.4), reviewed(1, 0.0, "good", 90.2)]
    assert any("White played more accurately (95% against 90%)" in line for line in narrative(far, "*"))
    assert any("similar accuracy (94% against 90%)" in line for line in narrative(near, "*"))


def test_narrative_with_a_draw_and_equal_play():
    moves = review_moves([ply(0, 0, 0), ply(1, 0, 0)])
    lines = narrative(moves, "1/2-1/2")
    assert lines[0] == "The game ended in a draw."
    assert any("similar accuracy" in line for line in lines)


def test_skipped_plies_are_not_judged():
    plies = [ply(0, 0, -300), ply(1, -300, -300), ply(2, -300, -300), ply(3, -300, -300)]
    moves = review_moves(plies, skip_plies=2)
    assert [move.category for move in moves[:2]] == ["random", "random"]
    assert moves[0].loss == 0.0
    assert moves[0].symbol == ""
    assert moves[2].category != "random"
    white = summarize_side(moves, True)
    assert white.moves == 1
    assert white.counts["blunder"] == 0


def test_skipped_plies_are_never_key_moments_or_decisive():
    moves = review_moves([ply(0, 0, 500), ply(1, 500, 500)], skip_plies=1)
    assert key_moments(moves) == []
    assert decisive_move(moves) is None


def test_the_narrative_ignores_skipped_errors():
    moves = review_moves([ply(0, 0, -300), ply(1, -300, -300)], skip_plies=1)
    assert not any("biggest error" in line for line in narrative(moves, "*"))


def test_nothing_is_skipped_by_default():
    assert all(move.category != "random" for move in review_moves([ply(0, 0, 0), ply(1, 0, 0)]))


def test_narrative_reports_a_game_decided_by_the_random_opening():
    moves = review_moves([ply(0, 0, 0), ply(1, 0, 900), ply(2, 900, 900)], skip_plies=2)
    lines = narrative(moves, "1-0")
    assert any("White was already winning before the first judged move (evaluation +9.00)" in line for line in lines)
    assert not any("Neither side" in line for line in lines)


def test_narrative_names_black_when_the_random_opening_favours_black():
    moves = review_moves([ply(0, 0, -900), ply(1, -900, -900)], skip_plies=1)
    assert any("Black was already winning" in line for line in narrative(moves, "0-1"))