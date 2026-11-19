from dataclasses import asdict

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from motifs.encoding import PLANES  # noqa: E402
from motifs.export import checkpoint_to_arrays, max_difference  # noqa: E402
from motifs.infer import MotifModel  # noqa: E402
from motifs.model import ModelConfig, build_model  # noqa: E402

CONFIG = ModelConfig(themes=3, channels=8, blocks=2, hidden=16)


def checkpoint() -> dict:
    torch.manual_seed(0)
    model = build_model(CONFIG)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 2.0)
            module.weight.data.uniform_(0.5, 1.5)
            module.bias.data.normal_()
    model.eval()
    return {
        "config": asdict(CONFIG),
        "themes": ["fork", "pin", "mateIn1"],
        "state_dict": model.state_dict(),
        "thresholds": [0.3, 0.4, 0.5],
    }


def test_the_numpy_network_matches_torch():
    saved = checkpoint()
    arrays = checkpoint_to_arrays(saved)
    planes = (np.random.default_rng(0).random((16, PLANES, 8, 8)) < 0.2).astype(np.uint8)
    assert max_difference(saved, arrays, planes) < 1e-4


def test_the_exported_model_keeps_themes_and_thresholds():
    model = MotifModel(checkpoint_to_arrays(checkpoint()))
    assert model.themes == ("fork", "pin", "mateIn1")
    assert np.allclose(model.thresholds, [0.3, 0.4, 0.5])
    assert model.blocks == 2