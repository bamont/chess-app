from chess.game import Game
from chess.notation import parse_uci
from chess.rules import generate_legal_moves
from ui.review_job import game_from_moves, played_moves


def legal(board, uci):
    return parse_uci(uci, generate_legal_moves(board, board.turn))


def test_a_played_game_becomes_a_pgn_game_with_san_moves():
    game = Game()
    for uci in ("e2e4", "e7e5", "g1f3"):
        game.play(legal(game.board, uci))
    pgn = game_from_moves(played_moves(game), {"White": "You"}, "*")
    assert pgn.moves == ["e4", "e5", "Nf3"]
    assert pgn.headers == {"White": "You"}
    assert pgn.result == "*"


def test_a_game_without_moves_has_no_san():
    assert game_from_moves(played_moves(Game()), {}, "*").moves == []


def test_captures_and_checks_are_written_in_san():
    game = Game()
    for uci in ("e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"):
        game.play(legal(game.board, uci))
    assert game_from_moves(played_moves(game), {}, "*").moves == ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5"]