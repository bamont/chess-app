import xml.etree.ElementTree as ET
from types import SimpleNamespace

from review.judge import key_moments, review_moves
from review.model import PlyAnalysis
from review.report import (
    board_svg,
    build_report,
    eval_graph,
    has_tactical_line,
    mate_in,
    render_report,
    theme_label,
    trusted_themes,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def ply(index: int, before: int, after: int, uci: str = "e2e4", best: str = "d2d4", line=("d4", "d5", "c4")) -> PlyAnalysis:
    return PlyAnalysis(index, START, "e4", uci, "d4", best, line, before, after)


def game(line=("d4", "d5", "c4")) -> list[PlyAnalysis]:
    return [
        ply(0, 30, 20, line=line),
        ply(1, 20, 20, line=line),
        ply(2, 20, -400, line=line),
        ply(3, -400, -420, line=line),
        ply(4, -420, -420, line=line),
    ]


def tactical_game() -> list[PlyAnalysis]:
    return game(line=("Bxd5", "exd5", "Qxd5"))


def test_board_has_every_square_and_piece():
    svg = board_svg(START)
    assert svg.count('class="sq-l"') + svg.count('class="sq-d"') == 64
    assert svg.count('class="piece ') == 32
    ET.fromstring(svg)


def test_flipping_the_board_shows_white_at_the_top():
    def first_piece(svg):
        return svg[svg.index('class="piece ') + len('class="piece '):][:7]

    assert first_piece(board_svg(START)) == "piece-b"
    assert first_piece(board_svg(START, flipped=True)) == "piece-w"


def test_the_pieces_of_a_position_are_drawn_from_its_fen():
    svg = board_svg("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert svg.count('class="piece ') == 2
    ET.fromstring(svg)


def test_both_moves_are_drawn_when_they_differ():
    svg = board_svg(START, "e2e4", "d2d4")
    assert 'class="arrow-played"' in svg
    assert 'class="arrow-best"' in svg
    ET.fromstring(svg)


def test_only_the_best_arrow_is_drawn_when_the_move_is_best():
    svg = board_svg(START, "e2e4", "e2e4")
    assert 'class="arrow-best"' in svg
    assert "arrow-played" not in svg


def test_invalid_moves_draw_no_arrow():
    svg = board_svg(START, "(none)", "(none)")
    assert "arrow" not in svg
    ET.fromstring(svg)


def test_promotion_moves_are_drawn_as_a_plain_arrow():
    svg = board_svg("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", "a7a8q")
    assert 'class="arrow-best"' in svg


def test_graph_marks_the_key_moments():
    moves = review_moves(game())
    svg = eval_graph(moves, key_moments(moves))
    assert "<polyline" in svg
    assert "dot-blunder" in svg
    ET.fromstring(svg)


def test_graph_without_moves_is_still_valid():
    svg = eval_graph([], [])
    assert "<svg" in svg
    ET.fromstring(svg)


def test_graph_with_one_move_is_valid():
    ET.fromstring(eval_graph(review_moves([ply(0, 0, 10)]), []))


def test_report_contains_every_section():
    report = build_report({"White": "Alice", "Black": "Bob", "Date": "2026.10.29"}, "0-1", game())
    for text in ("Alice vs Bob", "Accuracy", "Evaluation", "Key moments", "Moves", "2026.10.29", "0-1"):
        assert text in report
    assert report.startswith("<!DOCTYPE html>")
    assert report.count("<article") == 1


def test_report_lists_the_best_alternative_of_a_key_moment():
    report = build_report({}, "0-1", game())
    assert "Best was <strong>d4</strong>" in report
    assert "d4 d5 c4" in report


def test_report_escapes_header_values():
    report = build_report({"White": "<script>alert(1)</script>", "Black": 'x"y'}, "*", game())
    assert "<script>" not in report
    assert "&lt;script&gt;" in report


def test_report_without_mistakes_says_so():
    quiet = [ply(0, 0, 0), ply(1, 0, 0)]
    assert "No mistakes or blunders were found" in build_report({}, "1/2-1/2", quiet)


def test_report_uses_default_names():
    assert "White vs Black" in build_report({}, "*", game())


def test_render_report_accepts_a_single_move():
    report = render_report({}, "*", review_moves([ply(0, 0, 0)]))
    assert "<table class=\"moves\">" in report


def test_report_marks_random_opening_moves():
    report = build_report({}, "0-1", game(), skip_plies=2)
    assert "The first 2 plies were random opening moves and are not judged." in report
    assert report.count('<span class="alt">random</span>') == 2
    assert report.count("<article") == 1


def test_a_skipped_mistake_is_not_a_key_moment():
    report = build_report({}, "0-1", game(), skip_plies=3)
    assert "No mistakes or blunders were found" in report


def test_report_without_skipped_plies_has_no_note():
    assert "random opening moves" not in build_report({}, "0-1", game())


class FakeMotifs:
    def __init__(self, themes=(("fork", 0.8), ("pin", 0.6))):
        self.themes = themes
        self.requests = []

    def predict_fen(self, fen, limit=None):
        self.requests.append((fen, limit))
        return [SimpleNamespace(theme=theme, score=score) for theme, score in self.themes]


def test_theme_names_are_readable():
    assert theme_label("fork") == "fork"
    assert theme_label("discoveredAttack") == "discovered attack"
    assert theme_label("mateIn2") == "mate in 2"
    assert theme_label("backRankMate") == "back-rank mate"
    assert theme_label("xRayAttack") == "x-ray attack"
    assert theme_label("capturingDefender") == "capturing the defender"


def test_key_moments_show_the_suggested_themes():
    report = build_report({}, "0-1", tactical_game(), motif_model=FakeMotifs())
    assert "Possible tactical themes" in report
    assert ">fork</span>" in report
    assert ">pin</span>" in report
    assert 'title="model score 80%"' in report


def test_no_themes_are_shown_without_a_model():
    assert "Possible tactical themes" not in build_report({}, "0-1", game())


def test_only_key_moments_are_analysed():
    model = FakeMotifs()
    build_report({}, "0-1", tactical_game(), motif_model=model)
    assert len(model.requests) == 1
    assert model.requests[0] == (START, None)


def test_a_moment_without_themes_says_so():
    report = build_report({}, "0-1", tactical_game(), motif_model=FakeMotifs(themes=()))
    assert "No tactical theme stood out for this position." in report
    assert "Possible tactical themes" not in report


def test_no_remark_about_themes_without_a_model():
    assert "No tactical theme stood out" not in build_report({}, "0-1", game())


def test_a_key_moment_shows_its_position():
    assert f"<code>{START}</code>" in build_report({}, "0-1", game())


def test_theme_names_are_escaped():
    report = build_report({}, "0-1", tactical_game(), motif_model=FakeMotifs(themes=(("<b>x", 0.9),)))
    assert "<b>x" not in report
    assert "&lt;b&gt;x" in report


def test_a_quiet_engine_line_is_not_sent_to_the_model():
    model = FakeMotifs()
    report = build_report({}, "0-1", game(), motif_model=model)
    assert model.requests == []
    assert "Possible tactical themes" not in report
    assert "No tactical theme stood out" not in report


def test_lines_with_captures_checks_or_mates_are_tactical():
    assert has_tactical_line(["d4", "d5", "Nxd5"])
    assert has_tactical_line(["d4", "Bb5+"])
    assert has_tactical_line(["Qh7#"])
    assert not has_tactical_line(["d4", "d5", "c4", "e6"])
    assert not has_tactical_line([])


def test_only_the_first_plies_of_the_line_count():
    assert has_tactical_line(["d4", "d5", "c4", "dxc4"])
    assert not has_tactical_line(["d4", "d5", "c4", "e6", "Nxd5"])


def mating(before: int, index: int = 0) -> PlyAnalysis:
    return PlyAnalysis(index, START, "Qf7#", "d1f7", "Qxf7#", "d1f7", ("Qxf7#",), before, 0)


def first(plies: list[PlyAnalysis]):
    return review_moves(plies)[0]


def test_a_forced_mate_for_the_mover_is_read_from_the_engine_score():
    assert mate_in(first([mating(9980)])) == 2
    assert mate_in(first([mating(9990)])) == 1
    assert mate_in(first([mating(300)])) is None
    assert mate_in(first([mating(-9980)])) is None


def test_a_forced_mate_for_black_is_read_from_blacks_side():
    black_mover = review_moves([ply(0, 0, 0), mating(-9990, index=1)])[1]
    assert mate_in(black_mover) == 1
    assert mate_in(review_moves([ply(0, 0, 0), mating(9990, index=1)])[1]) is None


def predictions(*items):
    return [SimpleNamespace(theme=theme, score=score) for theme, score in items]


def test_mate_themes_from_the_network_are_never_trusted():
    mover = first([mating(9980)])
    kept = trusted_themes(mover, predictions(("mateIn2", 0.9), ("fork", 0.8)), 3)
    assert kept == [("fork", 0.8)]


def test_mate_patterns_need_a_mate_seen_by_the_engine():
    quiet = first([ply(0, 30, -400)])
    assert trusted_themes(quiet, predictions(("backRankMate", 0.9), ("pin", 0.7)), 3) == [("pin", 0.7)]
    mover = first([mating(9980)])
    assert trusted_themes(mover, predictions(("backRankMate", 0.9), ("smotheredMate", 0.8)), 3) == [
        ("backRankMate", 0.9),
        ("smotheredMate", 0.8),
    ]


def test_the_number_of_trusted_themes_is_limited():
    mover = first([ply(0, 30, -400)])
    kept = trusted_themes(mover, predictions(("fork", 0.9), ("pin", 0.8), ("skewer", 0.7), ("sacrifice", 0.6)), 3)
    assert [theme for theme, _ in kept] == ["fork", "pin", "skewer"]


def test_a_missed_mate_is_announced_without_any_model():
    report = build_report({}, "*", [mating(9980)])
    assert "Stockfish saw a forced mate in 2." in report


def test_no_mate_is_announced_without_one():
    assert "forced mate" not in build_report({}, "0-1", game())


def test_the_report_keeps_mate_patterns_only_when_the_engine_saw_a_mate():
    model = FakeMotifs(themes=(("mateIn1", 0.99), ("backRankMate", 0.9)))
    with_mate = build_report({}, "*", [mating(9990)], motif_model=model)
    assert ">back-rank mate</span>" in with_mate
    assert ">mate in 1</span>" not in with_mate
    without_mate = build_report({}, "0-1", tactical_game(), motif_model=model)
    assert "back-rank mate" not in without_mate
    assert "No tactical theme stood out" in without_mate