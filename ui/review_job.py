import shutil
import threading
import time
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ProgressCallback = Callable[[int, int], None]
Work = Callable[[ProgressCallback], Path]

RESULTS = {"WHITE_WINS": "1-0", "BLACK_WINS": "0-1", "ONGOING": "*"}
IDLE_MESSAGE = "Press A to review this game with the engine"
MISSING_ENGINE_MESSAGE = "Start with --engine PATH to enable the analysis"
TOO_SHORT_MESSAGE = "Play a few moves before asking for an analysis"
FINISHED_MESSAGE = "Analysis finished: R review, H report"
MINIMUM_PLIES = 2


class ReviewJob:
    def __init__(self, work: Work) -> None:
        self._work = work
        self._lock = threading.Lock()
        self._done = 0
        self._total = 0
        self._result: Path | None = None
        self._error: str | None = None
        self._finished = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def result(self) -> Path | None:
        with self._lock:
            return self._result

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def status(self) -> str:
        with self._lock:
            if self._error is not None:
                return f"Analysis failed: {self._error}"
            if self._finished:
                return "Analysis finished"
            if self._total:
                return f"Analysing {self._done}/{self._total}"
            return "Starting the analysis"

    def _progress(self, done: int, total: int) -> None:
        with self._lock:
            self._done = done
            self._total = total

    def _run(self) -> None:
        try:
            path = self._work(self._progress)
        except Exception as error:
            with self._lock:
                self._error = str(error) or type(error).__name__
                self._finished = True
            return
        with self._lock:
            self._result = path
            self._finished = True


def result_token(status: Any) -> str:
    return RESULTS.get(status.name, "1/2-1/2")


def played_moves(game: Any) -> list[Any]:
    return [entry.move for entry in game.board.history]


def game_from_moves(moves: Sequence[Any], headers: dict[str, str], result: str) -> Any:
    from chess.board import Board
    from chess.pgn import PgnGame
    from chess.san import move_to_san

    board = Board.starting_position()
    sans = []
    for move in moves:
        sans.append(move_to_san(board, move))
        board.make_move(move)
    return PgnGame(headers, sans, result)


def _default_analyse(game: Any, engine: str, depth: int, progress: ProgressCallback, lines: int = 1) -> Any:
    from review.analysis import analyse_game

    return analyse_game(game, (engine,), depth, progress=progress, lines=lines)


def _default_build(game: Any, plies: Any, model: Any) -> str:
    from review.report import build_report

    return build_report(game.headers, game.result, plies, 0, model)


def _default_model(path: str) -> Any:
    from motifs.infer import MotifModel

    return MotifModel.load(path)


def _default_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def make_work(
    game: Any,
    engine: str,
    depth: int,
    motifs: str | None,
    output_dir: Path,
    analyse: Callable[..., Any] = _default_analyse,
    build: Callable[..., str] = _default_build,
    load_model: Callable[[str], Any] = _default_model,
    opener: Callable[[str], object] | None = None,
    timestamp: Callable[[], str] = _default_timestamp,
    on_plies: Callable[[Any], None] | None = None,
    lines: int = 1,
) -> Work:
    def work(progress: ProgressCallback) -> Path:
        plies = analyse(game, engine, depth, progress, lines=lines)
        if on_plies is not None:
            on_plies(plies)
        model = load_model(motifs) if motifs else None
        html = build(game, plies, model)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"review-{timestamp()}.html"
        path.write_text(html, encoding="utf-8")
        if opener is not None:
            opener(path.resolve().as_uri())
        return path

    return work


def resolve_engine(value: str) -> str | None:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path)
    return shutil.which(value)


def player_names(human_color: Any, bot_name: str) -> tuple[str, str]:
    if human_color is None:
        return "White", "Black"
    return ("You", bot_name) if human_color.name == "WHITE" else (bot_name, "You")


class ReviewLauncher:
    def __init__(
        self,
        engine: str | None,
        depth: int = 12,
        motifs: str | None = None,
        output_dir: Path = Path("reviews"),
        build_game: Callable[..., Any] = game_from_moves,
        make: Callable[..., Work] = make_work,
        lines: int = 3,
        opener: Callable[[str], object] = webbrowser.open,
    ) -> None:
        self.engine = engine
        self.depth = depth
        self.motifs = motifs
        self.output_dir = output_dir
        self.lines = lines
        self._build_game = build_game
        self._make = make
        self._opener = opener
        self._lock = threading.Lock()
        self._job: ReviewJob | None = None
        self._notice: str | None = None
        self._plies: Any = None
        self._delivered = False
        self._generation = 0

    @property
    def available(self) -> bool:
        return self.engine is not None

    @property
    def busy(self) -> bool:
        return self._job is not None and not self._job.finished

    @property
    def plies(self) -> Any:
        with self._lock:
            return self._plies

    def take_fresh(self) -> Any:
        with self._lock:
            if self._plies is None or self._delivered:
                return None
            self._delivered = True
            return self._plies

    def open_report(self) -> bool:
        path = self._job.result if self._job is not None else None
        if path is None:
            return False
        self._opener(path.resolve().as_uri())
        return True

    def start(self, game: Any, status: Any, white: str, black: str) -> bool:
        self._notice = None
        if self.engine is None:
            self._notice = MISSING_ENGINE_MESSAGE
            return False
        if self.busy:
            return False
        moves = played_moves(game)
        if len(moves) < MINIMUM_PLIES:
            self._notice = TOO_SHORT_MESSAGE
            return False
        headers = {
            "Event": "Chess app game",
            "Date": time.strftime("%Y.%m.%d"),
            "White": white,
            "Black": black,
        }
        review = self._build_game(moves, headers, result_token(status))
        generation = self._generation
        work = self._make(
            review,
            self.engine,
            self.depth,
            self.motifs,
            self.output_dir,
            on_plies=lambda plies: self._store(plies, generation),
            lines=self.lines,
        )
        self._job = ReviewJob(work)
        self._job.start()
        return True

    def _store(self, plies: Any, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self._plies = plies
                self._delivered = False

    def wait(self, timeout: float | None = None) -> None:
        if self._job is not None:
            self._job.wait(timeout)

    def status(self) -> str:
        if self._job is not None:
            if self._job.finished and self._job.error is None:
                return FINISHED_MESSAGE
            return self._job.status()
        if self._notice is not None:
            return self._notice
        return IDLE_MESSAGE if self.available else MISSING_ENGINE_MESSAGE

    def reset(self) -> None:
        if not self.busy:
            self._job = None
        self._notice = None
        with self._lock:
            self._generation += 1
            self._plies = None
            self._delivered = False