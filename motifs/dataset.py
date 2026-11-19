import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motifs.encoding import PACKED_BYTES, apply_uci, encode_position, pack_positions, parse_fen

THEMES = (
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "doubleCheck",
    "xRayAttack",
    "hangingPiece",
    "trappedPiece",
    "capturingDefender",
    "attraction",
    "deflection",
    "clearance",
    "interference",
    "intermezzo",
    "sacrifice",
    "quietMove",
    "exposedKing",
    "advancedPawn",
    "promotion",
    "backRankMate",
    "smotheredMate",
    "mateIn1",
    "mateIn2",
    "mateIn3",
)
TRAIN, VALIDATION, TEST = 0, 1, 2
VALIDATION_SHARE = 5
TEST_SHARE = 5


@dataclass(frozen=True)
class PuzzleSet:
    positions: np.ndarray
    labels: np.ndarray
    ratings: np.ndarray
    splits: np.ndarray
    ids: np.ndarray
    themes: tuple[str, ...]
    negatives: np.ndarray


@dataclass(frozen=True)
class BuildReport:
    read: int
    kept: int
    skipped: int
    negatives: int
    theme_counts: dict[str, int]
    unseen_themes: tuple[str, ...]
    unknown_theme_counts: dict[str, int]


def split_of(puzzle_id: str) -> int:
    bucket = zlib.crc32(puzzle_id.encode("utf-8")) % 100
    if bucket < TEST_SHARE:
        return TEST
    if bucket < TEST_SHARE + VALIDATION_SHARE:
        return VALIDATION
    return TRAIN


def puzzle_position(fen: str, moves: str) -> tuple[list[list[str]], bool]:
    rows, white_to_move = parse_fen(fen)
    first = moves.split()[0]
    apply_uci(rows, first)
    return rows, not white_to_move


def build_dataset(
    records: Iterable[Mapping[str, str]],
    themes: Sequence[str] = THEMES,
    limit: int | None = None,
    with_negatives: bool = False,
) -> tuple[PuzzleSet, BuildReport]:
    theme_index = {name: index for index, name in enumerate(themes)}
    packed: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ratings: list[int] = []
    splits: list[int] = []
    ids: list[str] = []
    negative_flags: list[int] = []
    counts = {name: 0 for name in themes}
    other: dict[str, int] = {}
    read = skipped = 0
    for record in records:
        if limit is not None and read >= limit:
            break
        read += 1
        try:
            rows, white_to_move = puzzle_position(record["FEN"], record["Moves"])
            rating = int(record["Rating"])
        except (KeyError, ValueError, IndexError):
            skipped += 1
            continue
        label = np.zeros(len(themes), dtype=np.uint8)
        for theme in record.get("Themes", "").split():
            index = theme_index.get(theme)
            if index is None:
                other[theme] = other.get(theme, 0) + 1
            else:
                label[index] = 1
                counts[theme] += 1
        puzzle_id = record.get("PuzzleId", str(read))
        packed.append(pack_positions(encode_position(rows, white_to_move)[np.newaxis])[0])
        labels.append(label)
        ratings.append(rating)
        ids.append(puzzle_id)
        splits.append(split_of(puzzle_id))
        negative_flags.append(0)
        if with_negatives:
            before_rows, before_white_to_move = parse_fen(record["FEN"])
            packed.append(pack_positions(encode_position(before_rows, before_white_to_move)[np.newaxis])[0])
            labels.append(np.zeros(len(themes), dtype=np.uint8))
            ratings.append(rating)
            ids.append(f"{puzzle_id}n")
            splits.append(split_of(puzzle_id))
            negative_flags.append(1)
    if packed:
        positions = np.stack(packed)
        label_array = np.stack(labels)
    else:
        positions = np.zeros((0, PACKED_BYTES), dtype=np.uint8)
        label_array = np.zeros((0, len(themes)), dtype=np.uint8)
    dataset = PuzzleSet(
        positions,
        label_array,
        np.array(ratings, dtype=np.int16),
        np.array(splits, dtype=np.uint8),
        np.array(ids, dtype="U12"),
        tuple(themes),
        np.array(negative_flags, dtype=np.uint8),
    )
    unseen = tuple(name for name in themes if counts[name] == 0)
    kept = len(ids) - sum(negative_flags)
    return dataset, BuildReport(read, kept, skipped, sum(negative_flags), counts, unseen, other)


def save_dataset(path: str | Path, dataset: PuzzleSet) -> None:
    np.savez_compressed(
        path,
        positions=dataset.positions,
        labels=dataset.labels,
        ratings=dataset.ratings,
        splits=dataset.splits,
        ids=dataset.ids,
        themes=np.array(dataset.themes),
        negatives=dataset.negatives,
    )


def load_dataset(path: str | Path) -> PuzzleSet:
    with np.load(path) as data:
        return PuzzleSet(
            data["positions"],
            data["labels"],
            data["ratings"],
            data["splits"],
            data["ids"],
            tuple(str(name) for name in data["themes"]),
            data["negatives"] if "negatives" in data.files else np.zeros(len(data["ids"]), dtype=np.uint8),
        )