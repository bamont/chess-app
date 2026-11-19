from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motifs.encoding import encode_position, parse_fen


@dataclass(frozen=True)
class MotifPrediction:
    theme: str
    score: float
    threshold: float


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def conv3x3(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
    return np.einsum("ncijkl,ockl->noij", windows, weight, optimize=True) + bias[None, :, None, None]


def conv1x1(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.einsum("ncij,oc->noij", x, weight[:, :, 0, 0], optimize=True) + bias[None, :, None, None]


class MotifModel:
    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.arrays = {name: np.asarray(value) for name, value in arrays.items()}
        self.blocks = int(self.arrays["blocks"])
        self.themes = tuple(str(name) for name in self.arrays["themes"])
        self.thresholds = self.arrays["thresholds"].astype(np.float64)

    @classmethod
    def load(cls, path: str | Path) -> "MotifModel":
        with np.load(path) as data:
            return cls({name: data[name] for name in data.files})

    def scores(self, planes: np.ndarray) -> np.ndarray:
        a = self.arrays
        x = relu(conv3x3(planes.astype(np.float32), a["stem_weight"], a["stem_bias"]))
        for index in range(self.blocks):
            out = relu(conv3x3(x, a[f"block{index}_first_weight"], a[f"block{index}_first_bias"]))
            out = conv3x3(out, a[f"block{index}_second_weight"], a[f"block{index}_second_bias"])
            x = relu(out + x)
        x = relu(conv1x1(x, a["reduce_weight"], a["reduce_bias"]))
        hidden = relu(x.reshape(len(x), -1) @ a["hidden_weight"].T + a["hidden_bias"])
        logits = hidden @ a["output_weight"].T + a["output_bias"]
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))

    def all_scores(self, fen: str) -> list[MotifPrediction]:
        rows, white_to_move = parse_fen(fen)
        planes = encode_position(rows, white_to_move)[np.newaxis]
        scores = self.scores(planes)[0]
        predictions = [
            MotifPrediction(theme, float(score), float(threshold))
            for theme, score, threshold in zip(self.themes, scores, self.thresholds, strict=True)
        ]
        predictions.sort(key=lambda item: -item.score / max(item.threshold, 1e-9))
        return predictions

    def predict_fen(self, fen: str, limit: int | None = None) -> list[MotifPrediction]:
        predictions = [item for item in self.all_scores(fen) if item.score >= item.threshold]
        return predictions[:limit] if limit is not None else predictions