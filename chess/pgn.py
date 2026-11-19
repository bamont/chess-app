import re
from collections.abc import Iterable
from dataclasses import dataclass

from chess.board import Board
from chess.game import GameStatus
from chess.notation import parse_uci
from chess.rules import generate_legal_moves
from chess.san import move_to_san

LINE_WIDTH = 80
HEADER_PATTERN = re.compile(r'^\[(\w+)\s+"((?:[^"\\]|\\.)*)"\]\s*$')
COMMENT_PATTERN = re.compile(r"\{[^}]*\}|;[^\n]*")
NAG_PATTERN = re.compile(r"\$\d+")
VARIATION_PATTERN = re.compile(r"\([^()]*\)")
MOVE_NUMBER_PATTERN = re.compile(r"^\d+\.+")
RESULT_TOKENS = ("1-0", "0-1", "1/2-1/2", "*")


@dataclass(frozen=True)
class PgnGame:
    headers: dict[str, str]
    moves: list[str]
    result: str


def result_token(status: GameStatus) -> str:
    if status == GameStatus.WHITE_WINS:
        return "1-0"
    if status == GameStatus.BLACK_WINS:
        return "0-1"
    if status == GameStatus.ONGOING:
        return "*"
    return "1/2-1/2"


def uci_moves_to_san(uci_moves: Iterable[str]) -> list[str]:
    board = Board.starting_position()
    moves = []
    for text in uci_moves:
        move = parse_uci(text, generate_legal_moves(board, board.turn))
        moves.append(move_to_san(board, move))
        board.make_move(move)
    return moves


def format_pgn(headers: dict[str, str], san_moves: list[str], result: str) -> str:
    lines = [f'[{key} "{value.replace(chr(34), chr(92) + chr(34))}"]' for key, value in headers.items()]
    tokens = []
    for index, san in enumerate(san_moves):
        if index % 2 == 0:
            tokens.append(f"{index // 2 + 1}.")
        tokens.append(san)
    tokens.append(result)
    movetext = []
    current = ""
    for token in tokens:
        if current and len(current) + 1 + len(token) > LINE_WIDTH:
            movetext.append(current)
            current = token
        else:
            current = f"{current} {token}" if current else token
    if current:
        movetext.append(current)
    return "\n".join(lines + [""] + movetext) + "\n"


def _parse_movetext(text: str) -> tuple[list[str], str]:
    text = COMMENT_PATTERN.sub(" ", text)
    text = NAG_PATTERN.sub(" ", text)
    previous = None
    while previous != text:
        previous = text
        text = VARIATION_PATTERN.sub(" ", text)
    moves: list[str] = []
    result = "*"
    for token in text.split():
        if token in RESULT_TOKENS:
            result = token
            continue
        token = MOVE_NUMBER_PATTERN.sub("", token)
        if token:
            moves.append(token)
    return moves, result


def _build_game(headers: dict[str, str], lines: list[str]) -> PgnGame:
    moves, result = _parse_movetext("\n".join(lines))
    if result == "*":
        result = headers.get("Result", "*")
    return PgnGame(headers, moves, result)


def read_pgn(text: str) -> list[PgnGame]:
    games: list[PgnGame] = []
    headers: dict[str, str] = {}
    movetext: list[str] = []
    for line in text.splitlines():
        match = HEADER_PATTERN.match(line)
        if match:
            if movetext:
                games.append(_build_game(headers, movetext))
                headers, movetext = {}, []
            headers[match.group(1)] = match.group(2).replace('\\"', '"')
        elif line.strip():
            movetext.append(line)
    if headers or movetext:
        games.append(_build_game(headers, movetext))
    return games