from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    san: str
    uci: str
    white_cp: int
    line: tuple[str, ...]


@dataclass(frozen=True)
class PlyAnalysis:
    ply: int
    fen: str
    san: str
    uci: str
    best_san: str
    best_uci: str
    best_line: tuple[str, ...]
    white_cp_before: int
    white_cp_after: int
    candidates: tuple[Candidate, ...] = ()