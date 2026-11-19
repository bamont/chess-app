import random
import time
from collections.abc import Callable

from bot.evaluation import MATE_SCORE, PIECE_VALUES, evaluate
from bot.opening_book import OpeningBook
from chess.board import Board
from chess.move import Move
from chess.pieces import Color, Kind
from chess.rules import (
    generate_legal_captures,
    generate_legal_moves,
    has_insufficient_material,
    is_in_check,
)

INFINITY = MATE_SCORE + 1000
MATE_THRESHOLD = MATE_SCORE - 1000
MIN_REPETITION_CLOCK = 4
MAX_DEPTH = 100
MAX_PLY = 160
MAX_EXTENSION_PLY = 48
TT_LIMIT = 400_000
TT_MOVE_PRIORITY = 1_000_000
KILLER_PRIORITY = 5
LIMIT_CHECK_MASK = 63
NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2
LMR_MIN_DEPTH = 3
LMR_FIRST_REDUCED_MOVE = 3
LMR_DEEP_MOVE = 8
LMR_DEEP_DEPTH = 5
EXACT, LOWER, UPPER = 0, 1, 2

Evaluator = Callable[[Board], int]


def move_priority(board: Board, move: Move) -> int:
    priority = 0
    if move.captured is not None:
        attacker = board.piece_at(move.start)
        priority += 10 * PIECE_VALUES[move.captured.kind] - PIECE_VALUES[attacker.kind]
    if move.promotion is not None:
        priority += PIECE_VALUES[move.promotion]
    return priority


def order_moves(
    board: Board,
    moves: list[Move],
    tt_move: Move | None = None,
    killers: tuple | list = (),
) -> list[Move]:
    def key(move: Move) -> int:
        if tt_move is not None and move == tt_move:
            return TT_MOVE_PRIORITY
        priority = move_priority(board, move)
        if priority == 0 and killers and move in killers:
            return KILLER_PRIORITY
        return priority

    return sorted(moves, key=key, reverse=True)


def score_to_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


class MinimaxBot:
    def __init__(
        self,
        depth: int = 2,
        seed: int | None = None,
        pruning: bool = True,
        evaluator: Evaluator = evaluate,
        book: OpeningBook | None = None,
        quiescence_limit: int = 0,
        repetition_aware: bool = True,
        use_tt: bool = True,
        iterative: bool = True,
        time_limit: float | None = None,
        node_limit: int | None = None,
        null_move: bool = False,
        lmr: bool = False,
        check_extension: bool = False,
    ) -> None:
        if depth < 1:
            raise ValueError("Depth must be at least 1")
        if depth > MAX_DEPTH:
            raise ValueError(f"Depth cannot exceed {MAX_DEPTH}")
        if quiescence_limit < 0:
            raise ValueError("Quiescence limit cannot be negative")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("Time limit must be positive")
        if node_limit is not None and node_limit <= 0:
            raise ValueError("Node limit must be positive")
        if (time_limit is not None or node_limit is not None) and not iterative:
            raise ValueError("Time and node limits require iterative deepening")
        self.depth = depth
        self.pruning = pruning
        self.evaluator = evaluator
        self.book = book
        self.quiescence_limit = quiescence_limit
        self.repetition_aware = repetition_aware
        self.use_tt = use_tt
        self.iterative = iterative
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.null_move = null_move
        self.lmr = lmr
        self.check_extension = check_extension
        self.rng = random.Random(seed)
        self.nodes = 0
        self.qnodes = 0
        self.depth_reached = 0
        self.last_score: int | None = None
        self._limited = time_limit is not None or node_limit is not None
        self._stop = False
        self._deadline: float | None = None
        self._tt: dict[int, tuple] = {}
        self._killers: list[list[Move]] = []
        self._seen: set[int] = set()
        self._path: set[int] = set()

    def choose_move(self, board: Board) -> Move:
        if self.book is not None:
            book_move = self.book.choose(board, self.rng)
            if book_move is not None:
                self.last_score = None
                return book_move
        score, best_moves = self.search(board)
        self.last_score = score
        return self.rng.choice(best_moves)

    def search(self, board: Board) -> tuple[int, list[Move]]:
        moves = generate_legal_moves(board, board.turn)
        if not moves:
            raise ValueError("No legal moves available")
        self._begin(board)
        if not self.pruning:
            return self._search_root_plain(board, moves)
        ordered = order_moves(board, moves)
        depths = range(1, self.depth + 1) if self.iterative else [self.depth]
        best_score = -INFINITY
        best_moves: list[Move] = []
        for depth in depths:
            result = self._search_root(board, ordered, depth)
            if result is None:
                break
            best_score, best_moves = result
            self.depth_reached = depth
            ordered = best_moves + [move for move in ordered if move not in best_moves]
        return best_score, best_moves

    def _begin(self, board: Board) -> None:
        self.nodes = 0
        self.qnodes = 0
        self.depth_reached = 0
        self._stop = False
        self._deadline = time.perf_counter() + self.time_limit if self.time_limit is not None else None
        self._tt = {}
        self._killers = [[] for _ in range(MAX_PLY)]
        board.recompute_zobrist()
        self._seen = self._previous_positions(board)
        self._path = set()
        root_key = self._repetition_key(board)
        if root_key is not None:
            self._path.add(root_key)

    def _repetition_key(self, board: Board) -> int | None:
        if not self.repetition_aware or board.halfmove_clock < MIN_REPETITION_CLOCK:
            return None
        return board.zobrist

    def _previous_positions(self, board: Board) -> set[int]:
        positions: set[int] = set()
        if not self.repetition_aware:
            return positions
        scratch = board.copy()
        for _ in range(min(len(scratch.history), scratch.halfmove_clock)):
            scratch.undo_move()
            positions.add(scratch.zobrist)
        return positions

    @staticmethod
    def _has_non_pawn_material(board: Board, color: Color) -> bool:
        for row in board.grid:
            for piece in row:
                if (
                    piece is not None
                    and piece.color is color
                    and piece.kind is not Kind.PAWN
                    and piece.kind is not Kind.KING
                ):
                    return True
        return False

    def _check_limits(self) -> None:
        if self.depth_reached == 0:
            return
        if self.node_limit is not None and self.nodes + self.qnodes >= self.node_limit or self._deadline is not None and time.perf_counter() > self._deadline:
            self._stop = True

    def _record_killer(self, ply: int, move: Move) -> None:
        killers = self._killers[ply]
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]

    def _search_root(
        self, board: Board, moves: list[Move], depth: int
    ) -> tuple[int, list[Move]] | None:
        best_score = -INFINITY
        best_moves: list[Move] = []
        for move in moves:
            board.make_move(move)
            alpha = best_score - 1 if best_moves else -INFINITY
            score = -self._alphabeta(board, depth - 1, 1, -INFINITY, -alpha)
            board.undo_move()
            if self._stop:
                return None
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        return best_score, best_moves

    def _search_root_plain(self, board: Board, moves: list[Move]) -> tuple[int, list[Move]]:
        best_score = -INFINITY
        best_moves: list[Move] = []
        for move in moves:
            board.make_move(move)
            score = -self._negamax(board, self.depth - 1, 1)
            board.undo_move()
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        self.depth_reached = self.depth
        return best_score, best_moves

    def _leaf(self, board: Board, alpha: int, beta: int, ply: int, in_check: bool) -> int:
        if self.quiescence_limit == 0:
            if in_check and not generate_legal_moves(board, board.turn):
                return -(MATE_SCORE - ply)
            return self.evaluator(board)
        return self._quiescence(board, alpha, beta, 0, ply)

    def _leaf_full(self, board: Board, ply: int) -> int:
        if self.quiescence_limit == 0:
            if is_in_check(board, board.turn) and not generate_legal_moves(board, board.turn):
                return -(MATE_SCORE - ply)
            return self.evaluator(board)
        return self._quiescence_full(board, 0, ply)

    def _negamax(self, board: Board, depth: int, ply: int) -> int:
        self.nodes += 1
        key = self._repetition_key(board)
        if key is not None and (key in self._seen or key in self._path):
            return 0
        if board.halfmove_clock >= 100 or has_insufficient_material(board):
            return 0
        if depth == 0:
            return self._leaf_full(board, ply)
        moves = generate_legal_moves(board, board.turn)
        if not moves:
            return -(MATE_SCORE - ply) if is_in_check(board, board.turn) else 0
        if key is not None:
            self._path.add(key)
        best = -INFINITY
        for move in moves:
            board.make_move(move)
            best = max(best, -self._negamax(board, depth - 1, ply + 1))
            board.undo_move()
        if key is not None:
            self._path.discard(key)
        return best

    def _alphabeta(
        self,
        board: Board,
        depth: int,
        ply: int,
        alpha: int,
        beta: int,
        allow_null: bool = True,
    ) -> int:
        self.nodes += 1
        if self._stop:
            return 0
        if self._limited and (self.nodes & LIMIT_CHECK_MASK) == 0:
            self._check_limits()
            if self._stop:
                return 0
        key = self._repetition_key(board)
        if key is not None and (key in self._seen or key in self._path):
            return 0
        if board.halfmove_clock >= 100 or has_insufficient_material(board):
            return 0
        turn = board.turn
        in_check = is_in_check(board, turn)
        if in_check and self.check_extension and ply < MAX_EXTENSION_PLY:
            depth += 1
        if depth == 0:
            return self._leaf(board, alpha, beta, ply, in_check)
        zobrist = board.zobrist
        tt_move = None
        if self.use_tt:
            entry = self._tt.get(zobrist)
            if entry is not None:
                entry_depth, flag, entry_score, tt_move = entry
                if entry_depth >= depth:
                    score = score_from_tt(entry_score, ply)
                    if flag == EXACT or (flag == LOWER and score >= beta) or (flag == UPPER and score <= alpha):
                        return score
        if (
            self.null_move
            and allow_null
            and depth >= NULL_MOVE_MIN_DEPTH
            and beta < MATE_THRESHOLD
            and not in_check
            and self._has_non_pawn_material(board, turn)
        ):
            saved = board.make_null_move()
            score = -self._alphabeta(
                board, depth - 1 - NULL_MOVE_REDUCTION, ply + 1, -beta, -beta + 1, False
            )
            board.undo_null_move(saved)
            if self._stop:
                return 0
            if score >= beta:
                return beta
        moves = generate_legal_moves(board, turn)
        if not moves:
            return -(MATE_SCORE - ply) if in_check else 0
        original_alpha = alpha
        if key is not None:
            self._path.add(key)
        best = -INFINITY
        best_move = None
        for index, move in enumerate(order_moves(board, moves, tt_move, self._killers[ply])):
            quiet = move.captured is None and move.promotion is None
            board.make_move(move)
            reduction = 0
            if (
                self.lmr
                and quiet
                and not in_check
                and depth >= LMR_MIN_DEPTH
                and index >= LMR_FIRST_REDUCED_MOVE
                and not is_in_check(board, board.turn)
            ):
                reduction = 2 if index >= LMR_DEEP_MOVE and depth >= LMR_DEEP_DEPTH else 1
            if reduction:
                score = -self._alphabeta(board, depth - 1 - reduction, ply + 1, -alpha - 1, -alpha)
                if score > alpha and not self._stop:
                    score = -self._alphabeta(board, depth - 1, ply + 1, -beta, -alpha)
            else:
                score = -self._alphabeta(board, depth - 1, ply + 1, -beta, -alpha)
            board.undo_move()
            if self._stop:
                if key is not None:
                    self._path.discard(key)
                return 0
            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                if quiet:
                    self._record_killer(ply, move)
                break
        if key is not None:
            self._path.discard(key)
        if self.use_tt:
            if best <= original_alpha:
                flag, stored_move = UPPER, tt_move
            elif best >= beta:
                flag, stored_move = LOWER, best_move
            else:
                flag, stored_move = EXACT, best_move
            existing = self._tt.get(zobrist)
            if existing is None or existing[0] <= depth:
                if len(self._tt) >= TT_LIMIT:
                    self._tt.clear()
                self._tt[zobrist] = (depth, flag, score_to_tt(best, ply), stored_move)
        return best

    def _quiescence(self, board: Board, alpha: int, beta: int, qply: int, ply: int) -> int:
        self.qnodes += 1
        if qply > 0 and (board.halfmove_clock >= 100 or has_insufficient_material(board)):
            return 0
        turn = board.turn
        if is_in_check(board, turn):
            moves = generate_legal_moves(board, turn)
            if not moves:
                return -(MATE_SCORE - (ply + qply))
            if qply >= self.quiescence_limit:
                return self.evaluator(board)
            best = -INFINITY
        else:
            best = self.evaluator(board)
            if qply >= self.quiescence_limit or best >= beta:
                return best
            moves = generate_legal_captures(board, turn)
        if best > alpha:
            alpha = best
        for move in order_moves(board, moves):
            board.make_move(move)
            score = -self._quiescence(board, -beta, -alpha, qply + 1, ply)
            board.undo_move()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def _quiescence_full(self, board: Board, qply: int, ply: int) -> int:
        self.qnodes += 1
        if qply > 0 and (board.halfmove_clock >= 100 or has_insufficient_material(board)):
            return 0
        turn = board.turn
        if is_in_check(board, turn):
            moves = generate_legal_moves(board, turn)
            if not moves:
                return -(MATE_SCORE - (ply + qply))
            if qply >= self.quiescence_limit:
                return self.evaluator(board)
            best = -INFINITY
        else:
            best = self.evaluator(board)
            if qply >= self.quiescence_limit:
                return best
            moves = generate_legal_captures(board, turn)
        for move in moves:
            board.make_move(move)
            best = max(best, -self._quiescence_full(board, qply + 1, ply))
            board.undo_move()
        return best