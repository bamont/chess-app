import random

import pytest

from chess.board import Board
from chess.notation import parse_uci
from chess.pgn import format_pgn, read_pgn
from chess.rules import generate_legal_moves
from chess.san import move_to_san, parse_san

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/8/8/8/8/N1N1K3 w - - 0 1",
    "4k3/8/8/R7/8/8/8/R3K3 w - - 0 1",
    "8/7k/8/Q7/8/8/8/Q3Q1K1 w - - 0 1",
]

FEN_ROUND_TRIPS = POSITIONS + [
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/8/8/8/8/4K3 b - - 12 40",
]

SAMPLE_PGN = """[Event "A"]
[White "x"]
[Black "y"]
[Result "1-0"]

1. e4 {a comment} e5 $1 2. Nf3 (2. Nc3 Nf6) 2... Nc6 3. Bb5 1-0

[Event "B"]
[White "z"]
[Black "w"]

1. d4 d5 *
"""


@pytest.mark.parametrize("fen", FEN_ROUND_TRIPS)
def test_fen_round_trip(fen):
    fullmove = int(fen.split()[-1])
    assert Board.from_fen(fen).to_fen(fullmove) == fen


def test_fen_after_moves():
    board = Board.starting_position()
    board.make_move(parse_uci("e2e4", generate_legal_moves(board, board.turn)))
    assert board.to_fen(1) == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    board.make_move(parse_uci("e7e5", generate_legal_moves(board, board.turn)))
    assert board.to_fen(2) == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"


@pytest.mark.parametrize("fen", POSITIONS)
def test_parse_san_inverts_move_to_san(fen):
    board = Board.from_fen(fen)
    for move in generate_legal_moves(board, board.turn):
        assert parse_san(board, move_to_san(board, move)) == move


@pytest.mark.parametrize("seed", range(5))
def test_parse_san_along_random_games(seed):
    rng = random.Random(seed)
    board = Board.starting_position()
    for _ in range(40):
        moves = generate_legal_moves(board, board.turn)
        if not moves:
            break
        for move in rng.sample(moves, min(5, len(moves))):
            assert parse_san(board, move_to_san(board, move)) == move
        board.make_move(rng.choice(moves))


def test_parse_san_accepts_annotations_and_zero_castling():
    board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert parse_san(board, "O-O+").flag == "castle_k"
    assert parse_san(board, "0-0-0").flag == "castle_q"
    assert parse_san(board, "Rb1!?").end == (7, 1)


def test_parse_san_rejects_illegal_moves():
    board = Board.starting_position()
    with pytest.raises(ValueError):
        parse_san(board, "e5")
    with pytest.raises(ValueError):
        parse_san(board, "Nf6")


def test_pgn_round_trip():
    headers = {"Event": "Test", "White": 'say "hi"', "Black": "B", "Result": "1/2-1/2"}
    moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    games = read_pgn(format_pgn(headers, moves, "1/2-1/2"))
    assert len(games) == 1
    assert games[0].headers == headers
    assert games[0].moves == moves
    assert games[0].result == "1/2-1/2"


def test_reading_several_games_with_comments_and_variations():
    games = read_pgn(SAMPLE_PGN)
    assert len(games) == 2
    assert games[0].headers["White"] == "x"
    assert games[0].moves == ["e4", "e5", "Nf3", "Nc6", "Bb5"]
    assert games[0].result == "1-0"
    assert games[1].moves == ["d4", "d5"]
    assert games[1].result == "*"