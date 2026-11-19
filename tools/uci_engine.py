import re
import subprocess
from dataclasses import dataclass
from types import TracebackType

OPTION_PATTERN = re.compile(r"option name (.+?) type (\w+)(.*)")
MATE_CENTIPAWNS = 10_000


@dataclass(frozen=True)
class AnalysisLine:
    score_cp: int
    pv: tuple[str, ...] = ()


@dataclass(frozen=True)
class Analysis:
    best_move: str
    score_cp: int
    depth: int
    pv: tuple[str, ...] = ()
    lines: tuple[AnalysisLine, ...] = ()


def parse_option(line: str) -> tuple[str, int | None, int | None] | None:
    match = OPTION_PATTERN.match(line.strip())
    if match is None:
        return None
    name, _, rest = match.groups()
    minimum = re.search(r"\bmin (-?\d+)", rest)
    maximum = re.search(r"\bmax (-?\d+)", rest)
    return (
        name,
        int(minimum.group(1)) if minimum else None,
        int(maximum.group(1)) if maximum else None,
    )


def parse_info_score(line: str) -> tuple[int, int] | None:
    tokens = line.split()
    if not tokens or tokens[0] != "info" or "score" not in tokens:
        return None
    if "lowerbound" in tokens or "upperbound" in tokens:
        return None
    try:
        depth = int(tokens[tokens.index("depth") + 1]) if "depth" in tokens else 0
        position = tokens.index("score")
        kind, value = tokens[position + 1], int(tokens[position + 2])
    except (ValueError, IndexError):
        return None
    if kind == "cp":
        return depth, value
    if kind == "mate":
        if value == 0:
            return depth, -MATE_CENTIPAWNS
        sign = 1 if value > 0 else -1
        return depth, sign * (MATE_CENTIPAWNS - 10 * abs(value))
    return None


def parse_info_pv(line: str) -> tuple[str, ...]:
    tokens = line.split()
    if "pv" not in tokens:
        return ()
    return tuple(tokens[tokens.index("pv") + 1 :])


def parse_info_multipv(line: str) -> int:
    tokens = line.split()
    if "multipv" not in tokens:
        return 1
    try:
        return int(tokens[tokens.index("multipv") + 1])
    except (ValueError, IndexError):
        return 1


class UciEngine:
    def __init__(self, command: list[str]) -> None:
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._options: dict[str, tuple[int | None, int | None]] = {}
        self._line_count = 1
        self._send("uci")
        for line in self._read_until("uciok"):
            option = parse_option(line)
            if option is not None:
                name, minimum, maximum = option
                self._options[name] = (minimum, maximum)
        self._wait_ready()

    def __enter__(self) -> "UciEngine":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def has_option(self, name: str) -> bool:
        return name in self._options

    def option_range(self, name: str) -> tuple[int, int] | None:
        bounds = self._options.get(name)
        if bounds is None or bounds[0] is None or bounds[1] is None:
            return None
        return bounds

    def set_option(self, name: str, value: str) -> None:
        self._send(f"setoption name {name} value {value}")
        self._wait_ready()

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._wait_ready()

    def best_move(self, moves: list[str], movetime_ms: int) -> str:
        self._send(self._position_command(moves))
        self._send(f"go movetime {movetime_ms}")
        tokens = self._read_until("bestmove")[-1].split()
        if len(tokens) < 2:
            raise RuntimeError("The engine did not return a move")
        return tokens[1]

    def analyse(self, moves: list[str], depth: int, fen: str | None = None, lines: int = 1) -> Analysis:
        self._use_lines(lines)
        self._send(self._position_command(moves, fen))
        self._send(f"go depth {depth}")
        output = self._read_until("bestmove")
        tokens = output[-1].split()
        best_move = tokens[1] if len(tokens) > 1 else "(none)"
        latest: dict[int, tuple[int, int, tuple[str, ...]]] = {}
        for line in output:
            parsed = parse_info_score(line)
            if parsed is not None:
                latest[parse_info_multipv(line)] = (parsed[0], parsed[1], parse_info_pv(line))
        reached, score, pv = latest.get(1, (0, 0, ()))
        alternatives = tuple(
            AnalysisLine(line_score, line_pv)
            for _, (line_depth, line_score, line_pv) in sorted(latest.items())
            if line_depth == reached
        )
        return Analysis(best_move, score, reached, pv, alternatives)

    def close(self) -> None:
        process = self._process
        if process.poll() is None:
            try:
                self._send("quit")
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()

    def _use_lines(self, count: int) -> None:
        if count == self._line_count or not self.has_option("MultiPV"):
            return
        self.set_option("MultiPV", str(count))
        self._line_count = count

    @staticmethod
    def _position_command(moves: list[str], fen: str | None = None) -> str:
        base = f"position fen {fen}" if fen is not None else "position startpos"
        if moves:
            return f"{base} moves " + " ".join(moves)
        return base

    def _send(self, command: str) -> None:
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def _wait_ready(self) -> None:
        self._send("isready")
        self._read_until("readyok")

    def _read_until(self, prefix: str) -> list[str]:
        lines = []
        while True:
            line = self._process.stdout.readline()
            if line == "":
                raise RuntimeError("The engine closed unexpectedly")
            line = line.strip()
            lines.append(line)
            if line.startswith(prefix):
                return lines