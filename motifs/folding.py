from collections.abc import Mapping, Sequence

import numpy as np

BATCH_NORM_EPSILON = 1e-5


def fold_batch_norm(
    weight: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    epsilon: float = BATCH_NORM_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    scale = gamma / np.sqrt(variance + epsilon)
    return weight * scale[:, None, None, None], beta - mean * scale


def export_arrays(
    state: Mapping[str, np.ndarray],
    blocks: int,
    thresholds: Sequence[float],
    themes: Sequence[str],
) -> dict[str, np.ndarray]:
    def folded(conv: str, norm: str) -> tuple[np.ndarray, np.ndarray]:
        return fold_batch_norm(
            state[f"{conv}.weight"],
            state[f"{norm}.weight"],
            state[f"{norm}.bias"],
            state[f"{norm}.running_mean"],
            state[f"{norm}.running_var"],
        )

    arrays: dict[str, np.ndarray] = {}
    arrays["stem_weight"], arrays["stem_bias"] = folded("stem.0", "stem.1")
    for index in range(blocks):
        prefix = f"blocks.{index}"
        arrays[f"block{index}_first_weight"], arrays[f"block{index}_first_bias"] = folded(
            f"{prefix}.first", f"{prefix}.first_norm"
        )
        arrays[f"block{index}_second_weight"], arrays[f"block{index}_second_bias"] = folded(
            f"{prefix}.second", f"{prefix}.second_norm"
        )
    arrays["reduce_weight"], arrays["reduce_bias"] = folded("reduce.0", "reduce.1")
    arrays["hidden_weight"] = np.asarray(state["head.1.weight"])
    arrays["hidden_bias"] = np.asarray(state["head.1.bias"])
    arrays["output_weight"] = np.asarray(state["head.3.weight"])
    arrays["output_bias"] = np.asarray(state["head.3.bias"])
    arrays["thresholds"] = np.asarray(thresholds, dtype=np.float32)
    arrays["themes"] = np.array(list(themes))
    arrays["blocks"] = np.array(blocks)
    return {name: np.asarray(value, dtype=np.float32) if value.dtype.kind == "f" else value for name, value in arrays.items()}