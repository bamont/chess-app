from chess.notation import parse_uci
from chess.pieces import Color
from chess.rules import generate_legal_moves
from ui.cli import main


def run_cli(monkeypatch, capsys, inputs):
    answers = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    main()
    return capsys.readouterr().out


def test_fools_mate_ends_the_game(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, ["f2f3", "e7e5", "g2g4", "d8h4"])
    assert "Black wins" in out


def test_scholars_mate_ends_the_game(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"])
    assert "White wins" in out


def test_invalid_input_does_not_crash(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, ["bonjour", "e2e5", "", "quit"])
    assert "Invalid move format" in out
    assert "Illegal move" in out


def test_moves_command_lists_legal_moves(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, ["moves", "quit"])
    assert "e2e4" in out
    assert "g1f3" in out


def test_help_command(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, ["help", "quit"])
    assert out.count("Commands:") == 2


def test_eof_quits_cleanly(monkeypatch, capsys):
    def raise_eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    main()


class ScriptedBot:
    def __init__(self, *notations):
        self.notations = iter(notations)

    def choose_move(self, board):
        legal = generate_legal_moves(board, board.turn)
        return parse_uci(next(self.notations), legal)


def test_game_against_bot(monkeypatch, capsys):
    answers = iter(["f2f3", "g2g4"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    main(bot=ScriptedBot("e7e5", "d8h4"), human_color=Color.WHITE)
    out = capsys.readouterr().out
    assert "Bot plays e7e5" in out
    assert "Black wins" in out


def test_bot_plays_first_when_human_is_black(monkeypatch, capsys):
    answers = iter(["e7e5", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    main(bot=ScriptedBot("e2e4", "g1f3"), human_color=Color.BLACK)
    assert "Bot plays e2e4" in capsys.readouterr().out