import json
import sys
from pathlib import Path

from chess.board import Board
from chess.notation import move_to_uci
from chess.rules import generate_legal_moves
from tools.tactics import (
    PositionResult,
    SpecOutcome,
    TacticsTask,
    _line_san,
    format_details,
    format_report,
    format_unsolved,
    load_positions,
    refutation_kind,
    run_position,
)
from tools.uci_engine import UciEngine

FAKE_ENGINE = (sys.executable, str(Path(__file__).parent / "fake_uci_engine.py"))
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def outcome(spec: str, loss: int) -> SpecOutcome:
    return SpecOutcome(spec, "e2e4", "e4", loss, 0.5)


def refuted(refutation: tuple[str, ...], mated: bool = False, loss: int = 300) -> SpecOutcome:
    first = refutation[0] if refutation else "e4"
    return SpecOutcome("x", "e2e4", first, loss, 0.5, refutation, mated)


def test_analysis_can_start_from_a_fen():
    with UciEngine(list(FAKE_ENGINE)) as engine:
        analysis = engine.analyse([], 5, fen=KIWIPETE)
    board = Board.from_fen(KIWIPETE)
    legal = {move_to_uci(move) for move in generate_legal_moves(board, board.turn)}
    assert analysis.score_cp == 30
    assert analysis.best_move in legal


def test_analysis_from_a_fen_with_moves():
    with UciEngine(list(FAKE_ENGINE)) as engine:
        analysis = engine.analyse(["e2a6"], 5, fen=KIWIPETE)
    assert analysis.score_cp == 30


def test_each_bot_gets_a_loss_on_a_position():
    task = TacticsTask("game1-ply10", KIWIPETE, ("random", "greedy"), 5, FAKE_ENGINE, 0)
    result = run_position(task)
    assert result.id == "game1-ply10"
    assert result.fen == KIWIPETE
    assert [o.spec for o in result.outcomes] == ["random", "greedy"]
    assert all(o.loss == 60 for o in result.outcomes)
    assert all(len(o.move_uci) in (4, 5) for o in result.outcomes)
    assert result.best_san


def test_the_best_line_is_returned_in_san():
    result = run_position(TacticsTask("a", KIWIPETE, ("random",), 5, FAKE_ENGINE, 0))
    assert result.best_line == (result.best_san,)


def test_the_refutation_starts_with_the_played_move():
    result = run_position(TacticsTask("a", KIWIPETE, ("random",), 5, FAKE_ENGINE, 0))
    first = result.outcomes[0]
    assert first.refutation[0] == first.move_san
    assert len(first.refutation) == 2
    assert not first.mated


def test_line_san_follows_the_moves():
    assert _line_san(START, ("e2e4", "e7e5", "g1f3")) == ("e4", "e5", "Nf3")


def test_line_san_stops_at_an_illegal_move():
    assert _line_san(START, ("e2e4", "e2e4", "d7d5")) == ("e4",)


def test_line_san_respects_the_limit():
    assert _line_san(START, ("e2e4", "e7e5", "g1f3"), limit=2) == ("e4", "e5")


def test_positions_are_loaded_from_json_lines(tmp_path):
    path = tmp_path / "positions.jsonl"
    records = [{"id": "a", "fen": KIWIPETE}, {"id": "b", "fen": KIWIPETE}]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    assert [r["id"] for r in load_positions(str(path))] == ["a", "b"]
    assert [r["id"] for r in load_positions(str(path), limit=1)] == ["a"]


def test_report_counts_solved_positions():
    results = [
        PositionResult("a", "fen", "d4", (outcome("x", 10), outcome("y", 300))),
        PositionResult("b", "fen", "d4", (outcome("x", 150), outcome("y", 0))),
    ]
    text = format_report(results, ("x", "y"), 100)
    assert "2 positions" in text
    assert "x: solved 1/2 (50%), average loss 80 cp" in text
    assert "y: solved 1/2 (50%), average loss 150 cp" in text


def test_report_shows_which_bot_first_solves_each_position():
    results = [
        PositionResult("a", "fen", "d4", (outcome("x", 10), outcome("y", 300))),
        PositionResult("b", "fen", "d4", (outcome("x", 150), outcome("y", 0))),
        PositionResult("c", "fen", "d4", (outcome("x", 400), outcome("y", 500))),
    ]
    assert "first solved by: x 1, y 1, none 1" in format_report(results, ("x", "y"), 100)


def test_report_without_positions():
    assert format_report([], ("x",), 100) == "no positions"


def test_details_list_every_position():
    results = [PositionResult("a", "fen", "d4", (outcome("x", 10),))]
    assert format_details(results) == "a: best d4; x e4 (loss 10)"


def test_refutation_kinds():
    assert refutation_kind(refuted(("Qh5", "Qxh5"))) == "capture"
    assert refutation_kind(refuted(("Qh5", "Bb5+"))) == "check"
    assert refutation_kind(refuted(("Qh5", "a6"))) == "quiet"
    assert refutation_kind(refuted(("Qh5", "a6"), mated=True)) == "mate"
    assert refutation_kind(refuted(("Qh5", "Qxf7#"))) == "mate"
    assert refutation_kind(refuted(())) == "quiet"


def test_a_capture_with_check_counts_as_a_capture():
    assert refutation_kind(refuted(("Qh5", "Bxf7+"))) == "capture"


def test_unsolved_positions_are_listed_with_the_best_line():
    results = [
        PositionResult("a", "fen-a", "d4", (outcome("x", 10), outcome("y", 300))),
        PositionResult("b", "fen-b", "Qh5", (outcome("x", 400), outcome("y", 500)), ("Qh5", "g6", "Qxg6+")),
    ]
    text = format_unsolved(results, 100)
    assert "1 positions unsolved by every bot" in text
    assert "b: best line Qh5 g6 Qxg6+" in text
    assert "x e4 (400), y e4 (500)" in text
    assert "fen-b" in text
    assert "fen-a" not in text


def test_unsolved_falls_back_to_the_best_move_without_a_line():
    results = [PositionResult("b", "fen-b", "Qh5", (outcome("x", 400),))]
    assert "b: best line Qh5" in format_unsolved(results, 100)


def test_unsolved_listing_shows_the_refutation_and_a_summary():
    mated = SpecOutcome("y", "e2e4", "Qh5", 500, 0.5, ("Qh5", "g6", "Qxg6#"), True)
    results = [PositionResult("b", "fen-b", "Nf3", (outcome("x", 400), mated))]
    text = format_unsolved(results, 100)
    assert "refutation: Qh5 g6 Qxg6#" in text
    assert "mate 1, capture 0, check 0, quiet 0" in text


def test_unsolved_summary_counts_every_kind():
    results = [
        PositionResult("a", "fen-a", "d4", (refuted(("Qh5", "Qxh5")),)),
        PositionResult("b", "fen-b", "d4", (refuted(("Qh5", "Bb5+")),)),
        PositionResult("c", "fen-c", "d4", (refuted(("Qh5", "a6")),)),
        PositionResult("d", "fen-d", "d4", (refuted(("Qh5", "a6"), mated=True),)),
    ]
    assert "mate 1, capture 1, check 1, quiet 1" in format_unsolved(results, 100)


def test_unsolved_when_everything_is_solved():
    results = [PositionResult("a", "fen", "d4", (outcome("x", 10),))]
    assert format_unsolved(results, 100) == "every position is solved by at least one bot"