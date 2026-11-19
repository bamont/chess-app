from collections.abc import Callable

from chess.board import Board
from chess.notation import move_to_uci, parse_uci
from chess.pgn import PgnGame
from chess.rules import generate_legal_moves
from chess.san import move_to_san, parse_san
from review.model import Candidate, PlyAnalysis
from tools.uci_engine import Analysis, AnalysisLine, UciEngine

LINE_PLIES = 5
NO_MOVE = "(none)"


def line_to_san(fen: str, ucis: tuple[str, ...], limit: int = LINE_PLIES) -> tuple[str, ...]:
    board = Board.from_fen(fen)
    sans = []
    for uci in ucis[:limit]:
        try:
            move = parse_uci(uci, generate_legal_moves(board, board.turn))
        except ValueError:
            break
        sans.append(move_to_san(board, move))
        board.make_move(move)
    return tuple(sans)


def _white_centipawns(analysis: Analysis | AnalysisLine, ply: int) -> int:
    return analysis.score_cp if ply % 2 == 0 else -analysis.score_cp


def _candidates(fen: str, analysis: Analysis, ply: int) -> tuple[Candidate, ...]:
    candidates = []
    for line in analysis.lines:
        sans = line_to_san(fen, line.pv) if line.pv else ()
        if sans:
            candidates.append(Candidate(sans[0], line.pv[0], _white_centipawns(line, ply), sans))
    return tuple(candidates)


def analyse_game(
    game: PgnGame,
    engine_command: tuple[str, ...],
    depth: int,
    threads: int = 1,
    progress: Callable[[int, int], None] | None = None,
    lines: int = 1,
) -> list[PlyAnalysis]:
    if "FEN" in game.headers:
        raise ValueError("Games that start from a custom position are not supported")
    board = Board.starting_position()
    ucis: list[str] = []
    sans: list[str] = []
    fens: list[str] = []
    for text in game.moves:
        try:
            move = parse_san(board, text)
        except ValueError as error:
            raise ValueError(f"Cannot read move '{text}' at ply {len(ucis) + 1}: {error}") from error
        fens.append(board.to_fen(len(ucis) // 2 + 1))
        sans.append(move_to_san(board, move))
        ucis.append(move_to_uci(move))
        board.make_move(move)
    total = len(ucis)
    analyses: list[Analysis] = []
    with UciEngine(list(engine_command)) as engine:
        if threads > 1 and engine.has_option("Threads"):
            engine.set_option("Threads", str(threads))
        engine.new_game()
        for ply in range(total + 1):
            analyses.append(engine.analyse(ucis[:ply], depth, lines=lines))
            if progress is not None:
                progress(ply + 1, total + 1)
    plies = []
    for ply in range(total):
        before, after = analyses[ply], analyses[ply + 1]
        best_uci = before.best_move
        if best_uci == NO_MOVE:
            best_san, best_line = "-", ()
        else:
            best_san = line_to_san(fens[ply], (best_uci,), 1)[0]
            best_line = line_to_san(fens[ply], before.pv) if before.pv and before.pv[0] == best_uci else (best_san,)
        plies.append(
            PlyAnalysis(
                ply,
                fens[ply],
                sans[ply],
                ucis[ply],
                best_san,
                best_uci,
                best_line,
                _white_centipawns(before, ply),
                _white_centipawns(after, ply + 1),
                _candidates(fens[ply], before, ply),
            )
        )
    return plies