import pytest

np = pytest.importorskip("numpy")

from motifs.folding import export_arrays, fold_batch_norm  # noqa: E402
from motifs.infer import MotifModel, conv1x1, conv3x3  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_TO_MOVE = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
THEMES = ("fork", "pin", "mateIn1")
EPSILON = 1e-5


def naive_conv3x3(x, weight, bias):
    count, _, height, width = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((count, weight.shape[0], height, width))
    for item in range(count):
        for channel in range(weight.shape[0]):
            for row in range(height):
                for col in range(width):
                    window = padded[item, :, row : row + 3, col : col + 3]
                    out[item, channel, row, col] = (window * weight[channel]).sum() + bias[channel]
    return out


def naive_batch_norm(x, state, prefix):
    def column(name):
        return state[f"{prefix}.{name}"][None, :, None, None].astype(np.float64)

    return (x - column("running_mean")) / np.sqrt(column("running_var") + EPSILON) * column("weight") + column("bias")


def random_state(rng, channels=4, blocks=2, hidden=6, themes=3):
    state = {}

    def norm(prefix, size):
        state[f"{prefix}.weight"] = rng.uniform(0.5, 1.5, size)
        state[f"{prefix}.bias"] = rng.normal(size=size)
        state[f"{prefix}.running_mean"] = rng.normal(size=size)
        state[f"{prefix}.running_var"] = rng.uniform(0.5, 2.0, size)

    state["stem.0.weight"] = rng.normal(size=(channels, 12, 3, 3)) * 0.3
    norm("stem.1", channels)
    for index in range(blocks):
        for name in ("first", "second"):
            state[f"blocks.{index}.{name}.weight"] = rng.normal(size=(channels, channels, 3, 3)) * 0.3
            norm(f"blocks.{index}.{name}_norm", channels)
    state["reduce.0.weight"] = rng.normal(size=(8, channels, 1, 1)) * 0.3
    norm("reduce.1", 8)
    state["head.1.weight"] = rng.normal(size=(hidden, 8 * 64)) * 0.05
    state["head.1.bias"] = rng.normal(size=hidden)
    state["head.3.weight"] = rng.normal(size=(themes, hidden)) * 0.3
    state["head.3.bias"] = rng.normal(size=themes)
    return {name: value.astype(np.float32) for name, value in state.items()}


def reference_scores(state, planes, blocks):
    x = planes.astype(np.float64)
    zeros = np.zeros(state["stem.0.weight"].shape[0])
    x = np.maximum(naive_batch_norm(naive_conv3x3(x, state["stem.0.weight"], zeros), state, "stem.1"), 0)
    for index in range(blocks):
        prefix = f"blocks.{index}"
        out = naive_conv3x3(x, state[f"{prefix}.first.weight"], zeros)
        out = np.maximum(naive_batch_norm(out, state, f"{prefix}.first_norm"), 0)
        out = naive_conv3x3(out, state[f"{prefix}.second.weight"], zeros)
        out = naive_batch_norm(out, state, f"{prefix}.second_norm")
        x = np.maximum(out + x, 0)
    reduced = np.einsum("ncij,oc->noij", x, state["reduce.0.weight"][:, :, 0, 0].astype(np.float64))
    reduced = np.maximum(naive_batch_norm(reduced, state, "reduce.1"), 0)
    hidden = np.maximum(reduced.reshape(len(reduced), -1) @ state["head.1.weight"].T + state["head.1.bias"], 0)
    logits = hidden @ state["head.3.weight"].T + state["head.3.bias"]
    return 1.0 / (1.0 + np.exp(-logits))


def test_the_fast_convolution_matches_a_naive_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 8, 8))
    weight = rng.normal(size=(4, 3, 3, 3))
    bias = rng.normal(size=4)
    assert np.allclose(conv3x3(x, weight, bias), naive_conv3x3(x, weight, bias))


def test_the_one_by_one_convolution_mixes_channels_only():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 8, 8))
    weight = rng.normal(size=(5, 3, 1, 1))
    bias = rng.normal(size=5)
    expected = np.einsum("ncij,oc->noij", x, weight[:, :, 0, 0]) + bias[None, :, None, None]
    assert np.allclose(conv1x1(x, weight, bias), expected)


def test_folding_batch_norm_into_the_convolution():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 3, 8, 8))
    weight = rng.normal(size=(4, 3, 3, 3))
    state = {
        "n.weight": rng.uniform(0.5, 1.5, 4),
        "n.bias": rng.normal(size=4),
        "n.running_mean": rng.normal(size=4),
        "n.running_var": rng.uniform(0.5, 2.0, 4),
    }
    unfolded = naive_batch_norm(naive_conv3x3(x, weight, np.zeros(4)), state, "n")
    folded_weight, folded_bias = fold_batch_norm(
        weight, state["n.weight"], state["n.bias"], state["n.running_mean"], state["n.running_var"]
    )
    assert np.allclose(conv3x3(x, folded_weight, folded_bias), unfolded)


def test_the_exported_network_matches_the_unfolded_reference():
    rng = np.random.default_rng(3)
    state = random_state(rng)
    arrays = export_arrays(state, 2, [0.3, 0.4, 0.5], THEMES)
    planes = (rng.random((3, 12, 8, 8)) < 0.2).astype(np.uint8)
    scores = MotifModel(arrays).scores(planes)
    assert scores.shape == (3, 3)
    assert np.allclose(scores, reference_scores(state, planes, 2), atol=1e-4)


def test_the_exported_arrays_describe_the_model():
    arrays = export_arrays(random_state(np.random.default_rng(4)), 2, [0.3, 0.4, 0.5], THEMES)
    model = MotifModel(arrays)
    assert model.themes == THEMES
    assert model.blocks == 2
    assert np.allclose(model.thresholds, [0.3, 0.4, 0.5])


def test_predictions_keep_the_themes_above_their_threshold():
    state = random_state(np.random.default_rng(5))
    everything = MotifModel(export_arrays(state, 2, [0.0, 0.0, 0.0], THEMES)).predict_fen(START)
    assert sorted(prediction.theme for prediction in everything) == sorted(THEMES)
    assert MotifModel(export_arrays(state, 2, [2.0, 2.0, 2.0], THEMES)).predict_fen(START) == []


def test_predictions_are_ranked_by_how_far_they_pass_their_threshold():
    state = random_state(np.random.default_rng(6))
    predictions = MotifModel(export_arrays(state, 2, [0.1, 0.2, 0.3], THEMES)).predict_fen(START)
    ratios = [prediction.score / prediction.threshold for prediction in predictions]
    assert ratios == sorted(ratios, reverse=True)


def test_the_number_of_predictions_can_be_limited():
    state = random_state(np.random.default_rng(7))
    model = MotifModel(export_arrays(state, 2, [0.0, 0.0, 0.0], THEMES))
    assert len(model.predict_fen(START, limit=2)) == 2


def test_black_to_move_positions_are_accepted():
    state = random_state(np.random.default_rng(8))
    model = MotifModel(export_arrays(state, 2, [0.0, 0.0, 0.0], THEMES))
    assert len(model.predict_fen(BLACK_TO_MOVE)) == 3


def test_a_model_survives_saving(tmp_path):
    state = random_state(np.random.default_rng(9))
    arrays = export_arrays(state, 2, [0.3, 0.4, 0.5], THEMES)
    path = tmp_path / "motifs.npz"
    np.savez_compressed(path, **arrays)
    loaded = MotifModel.load(path)
    planes = (np.random.default_rng(10).random((2, 12, 8, 8)) < 0.2).astype(np.uint8)
    assert np.allclose(loaded.scores(planes), MotifModel(arrays).scores(planes))
    assert loaded.themes == THEMES


def test_all_scores_lists_every_theme_in_ranked_order():
    state = random_state(np.random.default_rng(11))
    model = MotifModel(export_arrays(state, 2, [0.9, 0.9, 0.9], THEMES))
    scores = model.all_scores(START)
    assert sorted(item.theme for item in scores) == sorted(THEMES)
    ratios = [item.score / item.threshold for item in scores]
    assert ratios == sorted(ratios, reverse=True)
    assert all(item.score >= item.threshold for item in model.predict_fen(START))


def test_the_score_table_marks_the_selected_themes():
    from motifs.explain import format_scores

    state = random_state(np.random.default_rng(12))
    selected = format_scores(MotifModel(export_arrays(state, 2, [0.0, 0.0, 0.0], THEMES)), START, 8)
    none = format_scores(MotifModel(export_arrays(state, 2, [2.0, 2.0, 2.0], THEMES)), START, 8)
    assert selected.count("yes") == 3
    assert "yes" not in none
    assert "fork" in none
    assert len(format_scores(MotifModel(export_arrays(state, 2, [0.0, 0.0, 0.0], THEMES)), START, 2).splitlines()) == 3


def test_selection_rates_count_positions_and_themes():
    from motifs.audit import selection_rates

    scores = np.array([[0.9, 0.1, 0.2], [0.8, 0.7, 0.1], [0.1, 0.1, 0.1], [0.2, 0.1, 0.6]])
    thresholds = np.array([0.5, 0.5, 0.5])
    report = selection_rates(scores, thresholds, THEMES)
    assert report.positions == 4
    assert report.any_theme == pytest.approx(0.75)
    assert report.theme_rates == {"fork": 0.5, "pin": 0.25, "mateIn1": 0.25}


def test_selection_rates_without_positions():
    from motifs.audit import selection_rates

    report = selection_rates(np.zeros((0, 3)), np.array([0.5, 0.5, 0.5]), THEMES)
    assert report.positions == 0
    assert report.any_theme == 0.0


def test_audit_over_fens_matches_the_model_selection():
    from motifs.audit import audit_fens

    state = random_state(np.random.default_rng(13))
    everything = MotifModel(export_arrays(state, 2, [0.0, 0.0, 0.0], THEMES))
    nothing = MotifModel(export_arrays(state, 2, [2.0, 2.0, 2.0], THEMES))
    assert audit_fens(everything, [START, BLACK_TO_MOVE]).any_theme == 1.0
    assert audit_fens(nothing, [START, BLACK_TO_MOVE]).any_theme == 0.0
    assert audit_fens(nothing, []).positions == 0


def test_the_audit_text_lists_the_most_frequent_themes():
    from motifs.audit import AuditReport, format_audit

    text = format_audit(AuditReport(10, 0.4, {"fork": 0.3, "pin": 0.1}), top=1)
    assert "10 positions" in text
    assert "40.0%" in text
    assert "fork" in text
    assert "pin" not in text


def test_examples_are_positions_where_a_theme_was_selected():
    from motifs.audit import select_examples

    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.1, 0.1], [0.8, 0.7, 0.1], [0.2, 0.1, 0.6]])
    thresholds = np.array([0.5, 0.5, 0.5])
    fens = ["a", "b", "c", "d"]
    examples = select_examples(scores, thresholds, THEMES, fens, count=10)
    assert sorted(fen for fen, _ in examples) == ["a", "c", "d"]
    by_fen = dict(examples)
    assert by_fen["c"] == [("fork", 0.8), ("pin", 0.7)]


def test_the_number_of_examples_is_limited_and_reproducible():
    from motifs.audit import select_examples

    scores = np.full((20, 3), 0.9)
    thresholds = np.array([0.5, 0.5, 0.5])
    fens = [str(index) for index in range(20)]
    first = select_examples(scores, thresholds, THEMES, fens, count=4, seed=1)
    again = select_examples(scores, thresholds, THEMES, fens, count=4, seed=1)
    assert len(first) == 4
    assert first == again


def test_no_example_when_nothing_is_selected():
    from motifs.audit import select_examples

    assert select_examples(np.zeros((3, 3)), np.array([0.5, 0.5, 0.5]), THEMES, ["a", "b", "c"], count=5) == []


def test_examples_are_printed_with_their_themes():
    from motifs.audit import format_examples

    text = format_examples([("fen-a", [("fork", 0.93), ("pin", 0.8)])])
    assert text == "fen-a\n  fork 0.93, pin 0.80"