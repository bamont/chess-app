import sys
import types

import pytest

from main import build_parser, run


def run_with(monkeypatch, arguments):
    received = {}
    gui = types.ModuleType("ui.gui")
    gui.main = lambda **options: received.update(options)
    monkeypatch.setitem(sys.modules, "ui.gui", gui)
    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])
    run()
    return received


def test_the_analysis_options_have_safe_defaults():
    arguments = build_parser().parse_args([])
    assert arguments.engine is None
    assert arguments.motifs is None
    assert arguments.analysis_depth == 12
    assert arguments.analysis_lines == 3


def test_the_analysis_options_are_read():
    arguments = build_parser().parse_args(
        ["--gui", "--engine", "stockfish", "--motifs", "m.npz", "--analysis-depth", "10", "--analysis-lines", "5"]
    )
    assert arguments.engine == "stockfish"
    assert arguments.motifs == "m.npz"
    assert arguments.analysis_depth == 10
    assert arguments.analysis_lines == 5


def test_the_graphical_interface_receives_the_analysis_options(monkeypatch, tmp_path):
    engine = tmp_path / "engine.exe"
    engine.write_text("x")
    model = tmp_path / "motifs.npz"
    model.write_text("x")
    received = run_with(
        monkeypatch,
        ["--gui", "--bot", "minimax:2", "--engine", str(engine), "--motifs", str(model), "--analysis-depth", "10", "--analysis-lines", "2"],
    )
    assert received["engine"] == str(engine)
    assert received["motifs"] == str(model)
    assert received["analysis_depth"] == 10
    assert received["analysis_lines"] == 2
    assert received["bot_name"] == "minimax:2"
    assert received["color_choice"] == "white"


def test_the_analysis_is_off_without_an_engine(monkeypatch):
    received = run_with(monkeypatch, ["--gui"])
    assert received["engine"] is None
    assert received["motifs"] is None
    assert received["bot_name"] == "Bot"
    assert received["analysis_lines"] == 3


def test_a_missing_engine_is_rejected(monkeypatch):
    with pytest.raises(SystemExit):
        run_with(monkeypatch, ["--gui", "--engine", "definitely-not-an-engine-xyz"])


def test_a_missing_motif_model_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_with(monkeypatch, ["--gui", "--motifs", str(tmp_path / "missing.npz")])


def test_the_analysis_options_need_the_graphical_interface(monkeypatch, tmp_path):
    engine = tmp_path / "engine.exe"
    engine.write_text("x")
    with pytest.raises(SystemExit):
        run_with(monkeypatch, ["--engine", str(engine)])