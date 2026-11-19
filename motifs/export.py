import argparse
import time

import numpy as np
import torch

from motifs.dataset import VALIDATION, load_dataset
from motifs.encoding import unpack_positions
from motifs.folding import export_arrays
from motifs.infer import MotifModel
from motifs.model import ModelConfig, build_model


def checkpoint_to_arrays(checkpoint: dict) -> dict[str, np.ndarray]:
    state = {name: value.detach().cpu().numpy() for name, value in checkpoint["state_dict"].items()}
    return export_arrays(state, checkpoint["config"]["blocks"], checkpoint["thresholds"], checkpoint["themes"])


def max_difference(checkpoint: dict, arrays: dict[str, np.ndarray], planes: np.ndarray) -> float:
    model = build_model(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(planes).float())).numpy()
    return float(np.abs(MotifModel(arrays).scores(planes) - expected).max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained checkpoint to a NumPy-only file and check it against torch.")
    parser.add_argument("checkpoint", help=".pt file written by motifs.train")
    parser.add_argument("output", help=".npz file to write")
    parser.add_argument("--dataset", help="dataset used to draw positions for the check")
    parser.add_argument("--check", type=int, default=2000, help="number of positions used to compare torch and NumPy")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint)
    arrays = checkpoint_to_arrays(checkpoint)
    np.savez_compressed(args.output, **arrays)
    if args.dataset:
        dataset = load_dataset(args.dataset)
        indices = np.flatnonzero(dataset.splits == VALIDATION)[: args.check]
        planes = unpack_positions(dataset.positions[indices])
    else:
        rng = np.random.default_rng(0)
        planes = (rng.random((args.check, 12, 8, 8)) < 0.1).astype(np.uint8)
    difference = max_difference(checkpoint, arrays, planes)
    started = time.perf_counter()
    MotifModel.load(args.output).scores(planes[:200])
    rate = min(200, len(planes)) / (time.perf_counter() - started)
    print(f"exported {len(arrays)} arrays to {args.output}")
    print(f"largest difference between torch and NumPy on {len(planes)} positions: {difference:.2e}")
    print(f"NumPy speed: {rate:.0f} positions/s")


if __name__ == "__main__":
    main()