import sys
from pathlib import Path

import pytest

from chess.pgn import PgnGame
from review.analysis import analyse_game, line_to_san

FAKE_ENGINE = (sys.executable, str(Path(__file__).parent / "fake_uci_engine.py"))
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def game(moves=("e4", "e5", "Nf3", "Nc6"), headers=None) -> PgnGame:
    return PgnGame(headers or {"White": "a", "Black": "b"}, list(moves), "*")


def test_every_move_gets_an_analysis():
    plies = analyse_game(game(), FAKE_ENGINE, 5)
    assert [item.ply for item in plies] == [0, 1, 2, 3]
    assert [item.san for item in plies] == ["e4", "e5", "Nf3", "Nc6"]
    assert [item.uci for item in plies] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert plies[0].fen == START
    assert plies[0].best_uci == "a2a3"
    assert plies[0].best_san == "a3"
    assert plies[0].best_line == ("a3",)


def test_scores_are_converted_to_whites_point_of_view():
    plies = analyse_game(game(), FAKE_ENGINE, 5)
    assert [item.white_cp_before for item in plies] == [30, -30, 30, -30]
    assert [item.white_cp_after for item in plies] == [-30, 30, -30, 30]


def test_progress_is_reported_for_every_position():
    calls = []
    analyse_game(game(), FAKE_ENGINE, 5, progress=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


def test_a_game_without_moves_has_nothing_to_analyse():
    assert analyse_game(game(moves=()), FAKE_ENGINE, 5) == []


def test_custom_start_positions_are_rejected():
    with pytest.raises(ValueError):
        analyse_game(game(headers={"FEN": START}), FAKE_ENGINE, 5)


def test_unreadable_moves_name_their_position():
    with pytest.raises(ValueError) as error:
        analyse_game(game(moves=("e4", "e5", "Ke3")), FAKE_ENGINE, 5)
    assert "ply 3" in str(error.value)


def test_line_to_san_follows_the_moves():
    assert line_to_san(START, ("e2e4", "e7e5", "g1f3")) == ("e4", "e5", "Nf3")


def test_line_to_san_stops_at_an_illegal_move():
    assert line_to_san(START, ("e2e4", "e2e4", "d7d5")) == ("e4",)


def test_line_to_san_respects_the_limit():
    assert line_to_san(START, ("e2e4", "e7e5", "g1f3"), limit=2) == ("e4", "e5")


def test_a_single_line_gives_one_candidate_equal_to_the_best_move():
    first = analyse_game(game(), FAKE_ENGINE, 5)[0]
    assert len(first.candidates) == 1
    assert first.candidates[0].uci == first.best_uci
    assert first.candidates[0].san == first.best_san


def test_several_lines_become_ranked_candidates():
    plies = analyse_game(game(), FAKE_ENGINE, 5, lines=3)
    first = plies[0]
    assert len(first.candidates) == 3
    assert first.candidates[0].uci == first.best_uci
    assert first.candidates[0].san == first.best_san
    assert [candidate.white_cp for candidate in first.candidates] == [30, 20, 10]
    assert len({candidate.uci for candidate in first.candidates}) == 3


def test_candidate_scores_are_given_from_whites_point_of_view():
    black_move = analyse_game(game(), FAKE_ENGINE, 5, lines=3)[1]
    assert [candidate.white_cp for candidate in black_move.candidates] == [-30, -20, -10]