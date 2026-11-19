import sys
import threading
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from ui.review_job import (
    FINISHED_MESSAGE,
    IDLE_MESSAGE,
    MISSING_ENGINE_MESSAGE,
    TOO_SHORT_MESSAGE,
    ReviewJob,
    ReviewLauncher,
    make_work,
    played_moves,
    player_names,
    resolve_engine,
    result_token,
)


class FakeStatus(Enum):
    ONGOING = 1
    WHITE_WINS = 2
    BLACK_WINS = 3
    STALEMATE = 4
    DRAW_FIFTY_MOVES = 5


class FakeColor(Enum):
    WHITE = 1
    BLACK = 2


def test_a_job_that_has_not_started_reports_nothing_done():
    job = ReviewJob(lambda progress: Path("x"))
    assert not job.finished
    assert job.result is None
    assert job.error is None
    assert job.status() == "Starting the analysis"


def test_a_finished_job_exposes_its_result(tmp_path):
    job = ReviewJob(lambda progress: tmp_path / "review.html")
    job.start()
    job.wait(5)
    assert job.finished
    assert job.result == tmp_path / "review.html"
    assert job.error is None
    assert job.status() == "Analysis finished"


def test_progress_is_visible_while_the_job_runs():
    release = threading.Event()
    reached = threading.Event()

    def work(progress):
        progress(3, 10)
        reached.set()
        release.wait(5)
        return Path("x")

    job = ReviewJob(work)
    job.start()
    assert reached.wait(5)
    assert not job.finished
    assert job.status() == "Analysing 3/10"
    release.set()
    job.wait(5)
    assert job.finished


def test_an_error_is_reported_instead_of_raised():
    def work(progress):
        raise RuntimeError("engine not found")

    job = ReviewJob(work)
    job.start()
    job.wait(5)
    assert job.finished
    assert job.result is None
    assert job.error == "engine not found"
    assert job.status() == "Analysis failed: engine not found"


def test_an_error_without_a_message_still_names_its_type():
    def work(progress):
        raise ValueError()

    job = ReviewJob(work)
    job.start()
    job.wait(5)
    assert job.error == "ValueError"


@pytest.mark.parametrize(
    "status, expected",
    [
        (FakeStatus.ONGOING, "*"),
        (FakeStatus.WHITE_WINS, "1-0"),
        (FakeStatus.BLACK_WINS, "0-1"),
        (FakeStatus.STALEMATE, "1/2-1/2"),
        (FakeStatus.DRAW_FIFTY_MOVES, "1/2-1/2"),
    ],
)
def test_the_game_status_becomes_a_pgn_result(status, expected):
    assert result_token(status) == expected


def test_the_played_moves_come_from_the_board_history():
    game = SimpleNamespace(board=SimpleNamespace(history=[SimpleNamespace(move="e2e4"), SimpleNamespace(move="e7e5")]))
    assert played_moves(game) == ["e2e4", "e7e5"]


def fake_work(tmp_path, calls, **overrides):
    game = SimpleNamespace(headers={"White": "a"}, result="*")

    def analyse(received, engine, depth, progress, lines=1):
        calls["analyse"] = (received, engine, depth, lines)
        calls.setdefault("order", []).append("analyse")
        progress(1, 2)
        progress(2, 2)
        return ["plies"]

    def build(received, plies, model):
        calls["build"] = (plies, model)
        calls.setdefault("order", []).append("build")
        return "<html>review</html>"

    arguments = dict(analyse=analyse, build=build, timestamp=lambda: "20260101-120000")
    arguments.update(overrides)
    return game, make_work(game, "stockfish.exe", 12, None, tmp_path / "reviews", **arguments)


def test_the_work_writes_the_report_without_opening_it_by_default(tmp_path):
    calls = {}
    game, work = fake_work(tmp_path, calls)
    seen = []
    path = work(lambda done, total: seen.append((done, total)))
    assert path == tmp_path / "reviews" / "review-20260101-120000.html"
    assert path.read_text(encoding="utf-8") == "<html>review</html>"
    assert calls["analyse"] == (game, "stockfish.exe", 12, 1)
    assert calls["build"] == (["plies"], None)
    assert seen == [(1, 2), (2, 2)]


def test_the_report_is_opened_when_an_opener_is_given(tmp_path):
    opened = []
    _, work = fake_work(tmp_path, {}, opener=opened.append)
    path = work(lambda done, total: None)
    assert opened == [path.resolve().as_uri()]


def test_the_analysed_moves_are_handed_over_before_the_report_is_built(tmp_path):
    calls = {}
    received = []

    def on_plies(plies):
        calls.setdefault("order", []).append("on_plies")
        received.append(plies)

    _, work = fake_work(tmp_path, calls, on_plies=on_plies)
    work(lambda done, total: None)
    assert received == [["plies"]]
    assert calls["order"] == ["analyse", "on_plies", "build"]


def test_the_number_of_lines_is_passed_to_the_analysis(tmp_path):
    calls = {}
    _, work = fake_work(tmp_path, calls, lines=3)
    work(lambda done, total: None)
    assert calls["analyse"][3] == 3


def test_the_motif_model_is_loaded_only_when_a_path_is_given(tmp_path):
    calls = {}
    work = make_work(
        SimpleNamespace(headers={}, result="*"),
        "stockfish.exe",
        12,
        "motifs.npz",
        tmp_path,
        analyse=lambda g, e, d, p, lines=1: ["plies"],
        build=lambda g, plies, model: f"model={model}",
        load_model=lambda path: calls.setdefault("path", path) and "loaded",
        timestamp=lambda: "t",
    )
    path = work(lambda done, total: None)
    assert calls["path"] == "motifs.npz"
    assert path.read_text(encoding="utf-8") == "model=loaded"


def test_an_engine_failure_surfaces_through_the_job(tmp_path):
    def analyse(game, engine, depth, progress, lines=1):
        raise FileNotFoundError("stockfish.exe")

    work = make_work(SimpleNamespace(headers={}, result="*"), "stockfish.exe", 12, None, tmp_path, analyse=analyse)
    job = ReviewJob(work)
    job.start()
    job.wait(5)
    assert job.error == "stockfish.exe"


def test_an_existing_engine_file_is_resolved(tmp_path):
    engine = tmp_path / "engine.exe"
    engine.write_text("x")
    assert resolve_engine(str(engine)) == str(engine)
    assert resolve_engine(sys.executable) == sys.executable


def test_a_missing_engine_is_not_resolved():
    assert resolve_engine("definitely-not-an-engine-xyz") is None


def test_player_names_follow_the_human_color():
    assert player_names(None, "minimax:3") == ("White", "Black")
    assert player_names(FakeColor.WHITE, "minimax:3") == ("You", "minimax:3")
    assert player_names(FakeColor.BLACK, "minimax:3") == ("minimax:3", "You")


def game_with(count):
    history = [SimpleNamespace(move=f"m{index}") for index in range(count)]
    return SimpleNamespace(board=SimpleNamespace(history=history))


def launcher_with(tmp_path, engine="stockfish.exe", hold=None, calls=None, opened=None):
    calls = calls if calls is not None else {}

    def build_game(moves, headers, result):
        calls["game"] = (moves, headers, result)
        return "pgn"

    def make(review, engine_path, depth, motifs, output_dir, on_plies=None, lines=1):
        calls["make"] = (review, engine_path, depth, motifs, output_dir, lines)

        def work(progress):
            progress(1, 1)
            if on_plies is not None:
                on_plies(["plies"])
            if hold is not None:
                hold.wait(5)
            return output_dir / "review.html"

        return work

    launcher = ReviewLauncher(
        engine,
        9,
        "motifs.npz",
        tmp_path,
        build_game=build_game,
        make=make,
        opener=(opened.append if opened is not None else (lambda uri: None)),
    )
    return launcher, calls


def test_without_an_engine_the_analysis_is_unavailable(tmp_path):
    launcher, _ = launcher_with(tmp_path, engine=None)
    assert not launcher.available
    assert launcher.status() == MISSING_ENGINE_MESSAGE
    assert not launcher.start(game_with(10), FakeStatus.ONGOING, "You", "Bot")
    assert launcher.status() == MISSING_ENGINE_MESSAGE


def test_a_new_launcher_invites_the_player_to_analyse(tmp_path):
    launcher, _ = launcher_with(tmp_path)
    assert launcher.available
    assert not launcher.busy
    assert launcher.status() == IDLE_MESSAGE
    assert launcher.plies is None
    assert launcher.take_fresh() is None


def test_a_game_that_is_too_short_is_refused(tmp_path):
    launcher, calls = launcher_with(tmp_path)
    assert not launcher.start(game_with(1), FakeStatus.ONGOING, "You", "Bot")
    assert launcher.status() == TOO_SHORT_MESSAGE
    assert "make" not in calls


def test_starting_an_analysis_builds_the_game_and_runs_the_job(tmp_path):
    launcher, calls = launcher_with(tmp_path)
    assert launcher.start(game_with(4), FakeStatus.WHITE_WINS, "You", "minimax:3")
    launcher.wait(5)
    moves, headers, result = calls["game"]
    assert moves == ["m0", "m1", "m2", "m3"]
    assert headers["White"] == "You"
    assert headers["Black"] == "minimax:3"
    assert headers["Event"] == "Chess app game"
    assert len(headers["Date"]) == 10
    assert result == "1-0"
    assert calls["make"] == ("pgn", "stockfish.exe", 9, "motifs.npz", tmp_path, 3)
    assert launcher.status() == FINISHED_MESSAGE
    assert not launcher.busy


def test_the_analysed_moves_are_delivered_once(tmp_path):
    launcher, _ = launcher_with(tmp_path)
    launcher.start(game_with(4), FakeStatus.ONGOING, "You", "Bot")
    launcher.wait(5)
    assert launcher.take_fresh() == ["plies"]
    assert launcher.take_fresh() is None
    assert launcher.plies == ["plies"]


def test_the_report_can_be_opened_once_it_exists(tmp_path):
    opened = []
    launcher, _ = launcher_with(tmp_path, opened=opened)
    assert not launcher.open_report()
    launcher.start(game_with(4), FakeStatus.ONGOING, "You", "Bot")
    launcher.wait(5)
    assert launcher.open_report()
    assert opened == [(tmp_path / "review.html").resolve().as_uri()]


def test_only_one_analysis_runs_at_a_time(tmp_path):
    release = threading.Event()
    launcher, calls = launcher_with(tmp_path, hold=release)
    assert launcher.start(game_with(4), FakeStatus.ONGOING, "You", "Bot")
    assert launcher.busy
    calls.clear()
    assert not launcher.start(game_with(4), FakeStatus.ONGOING, "You", "Bot")
    assert "make" not in calls
    release.set()
    launcher.wait(5)
    assert not launcher.busy


def test_resetting_keeps_a_running_analysis_but_clears_a_finished_one(tmp_path):
    release = threading.Event()
    launcher, _ = launcher_with(tmp_path, hold=release)
    launcher.start(game_with(4), FakeStatus.ONGOING, "You", "Bot")
    launcher.reset()
    assert launcher.busy
    release.set()
    launcher.wait(5)
    launcher.reset()
    assert launcher.status() == IDLE_MESSAGE
    assert launcher.plies is None


def test_the_moves_of_a_game_that_was_reset_are_not_delivered(tmp_path):
    release = threading.Event()
    launcher, _ = launcher_with(tmp_path, hold=release)
    launcher.start(game_with(4), FakeStatus.ONGOING, "You", "Bot")
    launcher.reset()
    release.set()
    launcher.wait(5)
    assert launcher.take_fresh() is None
    assert launcher.plies is None


def test_resetting_clears_a_notice(tmp_path):
    launcher, _ = launcher_with(tmp_path)
    launcher.start(game_with(1), FakeStatus.ONGOING, "You", "Bot")
    launcher.reset()
    assert launcher.status() == IDLE_MESSAGE


def test_a_failing_analysis_is_reported_by_the_launcher(tmp_path):
    def make(review, engine, depth, motifs, output_dir, on_plies=None, lines=1):
        def work(progress):
            raise RuntimeError("no engine")

        return work

    launcher = ReviewLauncher("x", 9, None, tmp_path, build_game=lambda *a: "pgn", make=make)
    launcher.start(game_with(4), FakeStatus.ONGOING, "You", "Bot")
    launcher.wait(5)
    assert launcher.status() == "Analysis failed: no engine"
    assert not launcher.busy
    assert launcher.take_fresh() is None