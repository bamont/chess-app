import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from motifs.dataset import TRAIN, build_dataset  # noqa: E402
from motifs.encoding import PLANES, pack_positions  # noqa: E402
from motifs.model import ModelConfig, build_model, parameter_count  # noqa: E402
from motifs.train import batch_tensor, predict, train  # noqa: E402

BISHOP_FEN = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SMALL = ModelConfig(themes=2, channels=8, blocks=1, hidden=16)


def records(count: int) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        if index % 2:
            rows.append({"PuzzleId": f"t{index}", "FEN": BISHOP_FEN, "Moves": "f2g3 e6e7", "Rating": "1500", "Themes": "fork"})
        else:
            rows.append({"PuzzleId": f"t{index}", "FEN": START, "Moves": "e2e4 e7e5", "Rating": "1200", "Themes": "pin"})
    return rows


def test_the_model_gives_one_logit_per_theme():
    model = build_model(SMALL)
    assert model(torch.zeros(3, PLANES, 8, 8)).shape == (3, 2)
    assert parameter_count(model) > 0


def test_packed_positions_become_float_tensors():
    rng = np.random.default_rng(0)
    planes = (rng.random((4, PLANES, 8, 8)) < 0.2).astype(np.uint8)
    tensor = batch_tensor(pack_positions(planes))
    assert tensor.shape == (4, PLANES, 8, 8)
    assert tensor.dtype == torch.float32
    assert np.array_equal(tensor.numpy().astype(np.uint8), planes)


def test_predictions_are_probabilities():
    model = build_model(SMALL)
    rng = np.random.default_rng(1)
    packed = pack_positions((rng.random((5, PLANES, 8, 8)) < 0.2).astype(np.uint8))
    scores = predict(model, packed, themes=2, batch_size=2)
    assert scores.shape == (5, 2)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0


def test_predicting_nothing_gives_an_empty_array():
    model = build_model(SMALL)
    assert predict(model, np.zeros((0, 96), dtype=np.uint8), themes=2).shape == (0, 2)


def test_training_writes_a_checkpoint_and_reports_every_theme(tmp_path):
    dataset, _ = build_dataset(records(400), ("fork", "pin"))
    output = tmp_path / "model.pt"
    results = train(dataset, SMALL, str(output), epochs=2, batch_size=32)
    assert [result.name for result in results] == ["fork", "pin"]
    checkpoint = torch.load(output)
    assert checkpoint["themes"] == ["fork", "pin"]
    assert len(checkpoint["thresholds"]) == 2
    assert checkpoint["config"]["channels"] == 8


def test_training_can_use_a_subset_of_the_puzzles(tmp_path):
    dataset, _ = build_dataset(records(400), ("fork", "pin"))
    assert int((dataset.splits == TRAIN).sum()) > 64
    train(dataset, SMALL, str(tmp_path / "model.pt"), epochs=1, batch_size=32, max_train=64)


def test_a_dataset_that_is_too_small_is_rejected(tmp_path):
    dataset, _ = build_dataset(records(10), ("fork", "pin"))
    with pytest.raises(ValueError):
        train(dataset, SMALL, str(tmp_path / "model.pt"), epochs=1, batch_size=256)