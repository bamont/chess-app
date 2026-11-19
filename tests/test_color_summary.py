from tools.stats import color_summary


def test_color_summary_counts_by_color():
    entries = [(True, 1.0), (True, 0.0), (False, 0.5), (False, 1.0)]
    assert color_summary(entries) == "as white: +1 =0 -1 (score 0.50), as black: +1 =1 -0 (score 0.75)"


def test_color_without_games_scores_zero():
    assert color_summary([(True, 1.0)]) == "as white: +1 =0 -0 (score 1.00), as black: +0 =0 -0 (score 0.00)"