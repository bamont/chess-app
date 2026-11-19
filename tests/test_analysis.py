import json
import sys
from pathlib import Path

from chess.pgn import uci_moves_to_san
from tools.analyze_games import (
    AnalysisTask,
    MoveReport,
    analyse_game,
    centipawn_loss,
    format_summary,
    format_worst,
    write_blunders,
)
from tools.uci_engine import Analysis, UciEngine, parse_info_pv, parse_info_score

FAKE_ENGINE = (sys.executable, str(Path(__file__).parent / "fake_uci_engine.py"))
GAME = ["e2e4", "e7e5", "g1f3", "b8c6"]


def task(colors=(True, False), skip=0) -> AnalysisTask:
    return AnalysisTask(1, tuple(uci_moves_to_san(GAME)), colors, skip, 5, FAKE_ENGINE)


def report(game: int, ply: int, loss: int) -> MoveReport:
    return MoveReport(game, ply, "e4", "e2e4", "d4", "d2d4", loss, "fen")


def test_centipawn_score_is_read():
    assert parse_info_score("info depth 12 seldepth 18 score cp -35 nodes 1234 pv e2e4") == (12, -35)


def test_mate_scores_are_converted():
    assert parse_info_score("info depth 9 score mate 3 pv e2e4") == (9, 9970)
    assert parse_info_score("info depth 9 score mate -2 pv e2e4") == (9, -9980)
    assert parse_info_score("info depth 0 score mate 0") == (0, -10000)


def test_bounds_and_other_lines_are_ignored():
    assert parse_info_score("info depth 5 score cp 10 lowerbound pv e2e4") is None
    assert parse_info_score("info depth 5 score cp 10 upperbound") is None
    assert parse_info_score("info string hello") is None
    assert parse_info_score("bestmove e2e4") is None


def test_loss_is_the_drop_of_the_movers_evaluation():
    assert centipawn_loss(100, 50) == 150
    assert centipawn_loss(30, -30) == 0
    assert centipawn_loss(100, -150) == 0


def test_loss_ignores_the_size_of_mate_scores():
    assert centipawn_loss(9000, 9000) == 2000


def test_analyse_with_the_fake_engine():
    with UciEngine(list(FAKE_ENGINE)) as engine:
        analysis = engine.analyse(["e2e4"], 5)
    assert isinstance(analysis, Analysis)
    assert analysis.score_cp == 30
    assert analysis.depth == 5
    assert len(analysis.best_move) in (4, 5)


def test_every_analysed_move_gets_a_report():
    reports = analyse_game(task())
    assert [r.ply for r in reports] == [0, 1, 2, 3]
    assert all(r.loss == 60 for r in reports)
    assert [r.played_uci for r in reports] == GAME
    assert reports[0].fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert reports[0].best_uci == "a2a3"
    assert reports[0].best_san == "a3"
    assert reports[0].played_san == "e4"


def test_only_the_requested_colors_are_analysed():
    assert [r.ply for r in analyse_game(task(colors=(True,)))] == [0, 2]
    assert [r.ply for r in analyse_game(task(colors=(False,)))] == [1, 3]


def test_skipped_plies_are_not_reported():
    assert [r.ply for r in analyse_game(task(skip=2))] == [2, 3]


def test_summary_counts_blunders_mistakes_and_phases():
    reports = [report(1, 4, 10), report(1, 30, 250), report(2, 30, 120), report(2, 70, 400), report(3, 6, 0)]
    text = format_summary(reports, 3, 200, 100)
    assert "3 games, 5 analysed moves, average centipawn loss 156" in text
    assert "opening: 2 moves" in text
    assert "middlegame: 2 moves" in text
    assert "late game: 1 moves" in text
    assert "blunders (>= 200 cp): 2 in 2 games" in text
    assert "mistakes (100-199 cp): 1" in text
    assert "first blunder of each game: median move 26" in text


def test_summary_without_moves():
    assert format_summary([], 0, 200, 100) == "no analysed moves"


def test_worst_moves_are_sorted_by_loss():
    text = format_worst([report(1, 10, 50), report(2, 10, 300), report(3, 10, 120)], 2)
    lines = text.splitlines()
    assert lines[0] == "worst moves:"
    assert "game 2" in lines[1] and "loss 300" in lines[1]
    assert "game 3" in lines[3] and "loss 120" in lines[3]
    assert "game 1" not in text


def test_blunders_are_written_as_json_lines(tmp_path):
    path = tmp_path / "blunders.jsonl"
    count = write_blunders([report(1, 10, 50), report(2, 12, 300)], 200, str(path))
    assert count == 1
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["id"] == "game2-ply12"
    assert records[0]["loss"] == 300
    assert records[0]["best_uci"] == "d2d4"


def test_principal_variation_is_read():
    assert parse_info_pv("info depth 12 score cp 20 pv e2e4 e7e5 g1f3") == ("e2e4", "e7e5", "g1f3")
    assert parse_info_pv("info depth 12 score cp 20") == ()


def test_analysis_carries_the_principal_variation():
    with UciEngine(list(FAKE_ENGINE)) as engine:
        analysis = engine.analyse(["e2e4"], 5)
    assert analysis.pv == (analysis.best_move,)