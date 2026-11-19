import sys
from pathlib import Path

from tools.uci_engine import UciEngine, parse_info_multipv

MULTIPV_ENGINE = [sys.executable, str(Path(__file__).parent / "fake_multipv_engine.py")]
PLAIN_ENGINE = [sys.executable, str(Path(__file__).parent / "fake_plain_engine.py")]


def test_the_line_number_is_read_from_an_info_line():
    assert parse_info_multipv("info depth 12 multipv 3 score cp 20 pv e2e4") == 3
    assert parse_info_multipv("info depth 12 score cp 20 pv e2e4") == 1
    assert parse_info_multipv("info multipv x score cp 20") == 1
    assert parse_info_multipv("info multipv") == 1


def test_a_single_line_is_returned_by_default():
    with UciEngine(MULTIPV_ENGINE) as engine:
        analysis = engine.analyse([], 3)
    assert len(analysis.lines) == 1
    assert analysis.best_move == "e2e4"
    assert analysis.score_cp == 40
    assert analysis.pv == ("e2e4", "e7e5")
    assert analysis.lines[0].score_cp == 40


def test_several_lines_are_returned_best_first():
    with UciEngine(MULTIPV_ENGINE) as engine:
        analysis = engine.analyse([], 3, lines=3)
    assert [line.score_cp for line in analysis.lines] == [40, 30, 20]
    assert [line.pv[0] for line in analysis.lines] == ["e2e4", "d2d4", "g1f3"]
    assert analysis.score_cp == 40
    assert analysis.depth == 3


def test_only_lines_from_the_final_depth_are_kept():
    with UciEngine(MULTIPV_ENGINE) as engine:
        analysis = engine.analyse([], 3, lines=4)
    assert len(analysis.lines) == 3


def test_the_engine_goes_back_to_one_line_when_asked():
    with UciEngine(MULTIPV_ENGINE) as engine:
        assert len(engine.analyse([], 3, lines=3).lines) == 3
        assert len(engine.analyse([], 3, lines=1).lines) == 1
        assert len(engine.analyse([], 3, lines=2).lines) == 2


def test_an_engine_without_several_lines_still_answers():
    with UciEngine(PLAIN_ENGINE) as engine:
        analysis = engine.analyse([], 4, lines=3)
    assert analysis.best_move == "e2e4"
    assert len(analysis.lines) == 1
    assert analysis.lines[0].pv == ("e2e4", "e7e5")