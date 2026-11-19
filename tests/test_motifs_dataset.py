import pytest

np = pytest.importorskip("numpy")

from motifs.build import format_report, read_records  # noqa: E402
from motifs.dataset import (  # noqa: E402
    TEST,
    THEMES,
    TRAIN,
    VALIDATION,
    build_dataset,
    load_dataset,
    puzzle_position,
    save_dataset,
    split_of,
)
from motifs.encoding import PACKED_BYTES, unpack_positions  # noqa: E402

BISHOP_FEN = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def record(puzzle_id="p1", fen=BISHOP_FEN, moves="f2g3 e6e7", rating="1500", themes="fork middlegame"):
    return {"PuzzleId": puzzle_id, "FEN": fen, "Moves": moves, "Rating": rating, "Themes": themes}


def test_the_first_move_is_played_before_the_puzzle_starts():
    rows, white_to_move = puzzle_position(BISHOP_FEN, "f2g3 e6e7")
    assert white_to_move
    assert rows[5][6] == "b"
    assert rows[6][5] == "."


def test_the_split_is_stable_and_roughly_proportional():
    assert split_of("00008") == split_of("00008")
    shares = [split_of(f"id{i}") for i in range(10000)]
    assert 0.85 < shares.count(TRAIN) / len(shares) < 0.95
    assert 0.02 < shares.count(VALIDATION) / len(shares) < 0.08
    assert 0.02 < shares.count(TEST) / len(shares) < 0.08


def test_records_become_positions_labels_and_ratings():
    records = [
        record("a", themes="fork pin middlegame"),
        record("b", fen=START, moves="e2e4 e7e5", rating="900", themes="mateIn1 short"),
    ]
    dataset, report = build_dataset(records, ("fork", "pin", "mateIn1"))
    assert dataset.positions.shape == (2, PACKED_BYTES)
    assert dataset.labels.tolist() == [[1, 1, 0], [0, 0, 1]]
    assert dataset.ratings.tolist() == [1500, 900]
    assert dataset.ids.tolist() == ["a", "b"]
    assert dataset.themes == ("fork", "pin", "mateIn1")
    assert int(unpack_positions(dataset.positions)[1].sum()) == 32
    assert report.theme_counts == {"fork": 1, "pin": 1, "mateIn1": 1}
    assert report.unknown_theme_counts == {"middlegame": 1, "short": 1}
    assert report.unseen_themes == ()


def test_unreadable_records_are_counted_and_skipped():
    records = [
        record("a"),
        record("b", fen="not a fen"),
        record("c", moves=""),
        record("d", rating="high"),
        {"PuzzleId": "e"},
        record("f", moves="e5e6"),
    ]
    dataset, report = build_dataset(records, ("fork",))
    assert report.read == 6
    assert report.kept == 1
    assert report.skipped == 5
    assert dataset.ids.tolist() == ["a"]


def test_themes_that_never_occur_are_reported():
    _, report = build_dataset([record(themes="fork")], ("fork", "pin"))
    assert report.unseen_themes == ("pin",)


def test_the_limit_stops_reading():
    dataset, report = build_dataset([record(str(i)) for i in range(10)], ("fork",), limit=4)
    assert len(dataset.ids) == 4
    assert report.read == 4


def test_an_empty_input_gives_empty_arrays():
    dataset, report = build_dataset([], ("fork", "pin"))
    assert dataset.positions.shape == (0, PACKED_BYTES)
    assert dataset.labels.shape == (0, 2)
    assert report.kept == 0


def test_negative_positions_come_from_before_the_first_move():
    dataset, report = build_dataset([record("a", themes="fork"), record("b", themes="pin")], ("fork", "pin"), with_negatives=True)
    assert dataset.ids.tolist() == ["a", "an", "b", "bn"]
    assert dataset.negatives.tolist() == [0, 1, 0, 1]
    assert dataset.labels.tolist() == [[1, 0], [0, 0], [0, 1], [0, 0]]
    assert report.kept == 2
    assert report.negatives == 2
    assert dataset.splits[0] == dataset.splits[1]
    assert not np.array_equal(dataset.positions[0], dataset.positions[1])


def test_no_negatives_are_added_by_default():
    dataset, report = build_dataset([record("a")], ("fork",))
    assert dataset.negatives.tolist() == [0]
    assert report.negatives == 0


def test_negatives_survive_saving_and_old_files_still_load(tmp_path):
    dataset, _ = build_dataset([record("a")], ("fork",), with_negatives=True)
    path = tmp_path / "puzzles.npz"
    save_dataset(path, dataset)
    assert load_dataset(path).negatives.tolist() == [0, 1]
    old = tmp_path / "old.npz"
    np.savez_compressed(
        old,
        positions=dataset.positions,
        labels=dataset.labels,
        ratings=dataset.ratings,
        splits=dataset.splits,
        ids=dataset.ids,
        themes=np.array(dataset.themes),
    )
    assert load_dataset(old).negatives.tolist() == [0, 0]


def test_the_report_mentions_the_negative_positions():
    _, report = build_dataset([record("a")], ("fork",), with_negatives=True)
    assert "added 1 positions from before the first move" in format_report(report)


def test_the_default_themes_have_no_duplicates():
    assert len(set(THEMES)) == len(THEMES)


def test_a_dataset_survives_saving(tmp_path):
    dataset, _ = build_dataset([record("a"), record("b", themes="pin")], ("fork", "pin"))
    path = tmp_path / "puzzles.npz"
    save_dataset(path, dataset)
    loaded = load_dataset(path)
    assert np.array_equal(loaded.positions, dataset.positions)
    assert np.array_equal(loaded.labels, dataset.labels)
    assert np.array_equal(loaded.ratings, dataset.ratings)
    assert np.array_equal(loaded.splits, dataset.splits)
    assert loaded.ids.tolist() == ["a", "b"]
    assert loaded.themes == ("fork", "pin")


def test_the_csv_is_read_with_its_header(tmp_path):
    path = tmp_path / "puzzles.csv"
    path.write_text(
        "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n"
        f"00008,{BISHOP_FEN},f2g3 e6e7,1900,76,95,8000,crushing hangingPiece long middlegame,https://example.org/x,\n",
        encoding="utf-8",
    )
    records = list(read_records(str(path)))
    assert records[0]["PuzzleId"] == "00008"
    dataset, report = build_dataset(records, ("hangingPiece",))
    assert dataset.labels.tolist() == [[1]]
    assert report.unknown_theme_counts["crushing"] == 1


def test_the_report_lists_counts_and_warnings():
    _, report = build_dataset([record(themes="fork crushing")], ("fork", "pin"))
    text = format_report(report)
    assert "read 1 puzzles, kept 1, skipped 0" in text
    assert "fork: 1" in text
    assert "themes never seen (check the spelling): pin" in text
    assert "crushing (1)" in text