import argparse
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from motifs.dataset import TEST, TRAIN, VALIDATION, PuzzleSet, load_dataset
from motifs.encoding import unpack_positions
from motifs.metrics import ThemeResult, evaluate_scores, format_results, mean_average_precision
from motifs.model import ModelConfig, MotifNet, build_model, parameter_count

SPEED_REPORT_STEP = 50


def batch_tensor(packed: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(unpack_positions(packed)).float()


def predict(model: MotifNet, packed: np.ndarray, themes: int, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(packed), batch_size):
            logits = model(batch_tensor(packed[start : start + batch_size]))
            outputs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0, themes), dtype=np.float32)


def evaluate(
    model: MotifNet,
    dataset: PuzzleSet,
    indices: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[ThemeResult]:
    scores = predict(model, dataset.positions[indices], len(dataset.themes))
    return evaluate_scores(dataset.labels[indices], scores, dataset.themes, thresholds)


def save_checkpoint(path: str, model: MotifNet, config: ModelConfig, dataset: PuzzleSet, results: list[ThemeResult]) -> None:
    torch.save(
        {
            "config": asdict(config),
            "themes": list(dataset.themes),
            "state_dict": model.state_dict(),
            "thresholds": [result.threshold for result in results],
            "validation_ap": [result.average_precision for result in results],
        },
        path,
    )


def train(
    dataset: PuzzleSet,
    config: ModelConfig,
    output: str,
    epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 2e-3,
    max_train: int | None = None,
    seed: int = 0,
) -> list[ThemeResult]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_ids = np.flatnonzero(dataset.splits == TRAIN)
    if max_train is not None:
        train_ids = rng.permutation(train_ids)[:max_train]
    validation_ids = np.flatnonzero(dataset.splits == VALIDATION)
    test_ids = np.flatnonzero(dataset.splits == TEST)
    steps_per_epoch = len(train_ids) // batch_size
    if steps_per_epoch == 0:
        raise ValueError("Not enough training puzzles for one batch")
    if len(validation_ids) == 0:
        raise ValueError("The dataset has no validation puzzles")

    model = build_model(config)
    print(f"{len(train_ids)} training, {len(validation_ids)} validation, {len(test_ids)} test puzzles; {parameter_count(model)} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=epochs * steps_per_epoch)
    loss_function = nn.BCEWithLogitsLoss()
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_ids)
        total_loss = 0.0
        started = time.perf_counter()
        for step in range(steps_per_epoch):
            batch = np.sort(order[step * batch_size : (step + 1) * batch_size])
            inputs = batch_tensor(dataset.positions[batch])
            targets = torch.from_numpy(dataset.labels[batch]).float()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if epoch == 1 and step + 1 == SPEED_REPORT_STEP:
                rate = SPEED_REPORT_STEP * batch_size / (time.perf_counter() - started)
                print(f"speed: {rate:.0f} puzzles/s, about {len(train_ids) / rate / 60:.1f} min per epoch")
        results = evaluate(model, dataset, validation_ids)
        mean_ap = mean_average_precision(results)
        print(f"epoch {epoch}/{epochs}: loss {total_loss / steps_per_epoch:.4f}, validation mean AP {mean_ap:.3f}, {time.perf_counter() - started:.0f}s")
        if mean_ap > best:
            best = mean_ap
            save_checkpoint(output, model, config, dataset, results)
    checkpoint = torch.load(output)
    model.load_state_dict(checkpoint["state_dict"])
    if len(test_ids):
        print("test set, thresholds chosen on the validation set")
        final = evaluate(model, dataset, test_ids, checkpoint["thresholds"])
    else:
        final = evaluate(model, dataset, validation_ids)
    print(format_results(final))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the motif classifier on a dataset built by motifs.build.")
    parser.add_argument("dataset", help=".npz file written by motifs.build")
    parser.add_argument("output", help="checkpoint file to write (.pt)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-train", type=int, help="only use this many training puzzles")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--threads", type=int, help="CPU threads used by torch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dataset = load_dataset(args.dataset)
    config = ModelConfig(len(dataset.themes), args.channels, args.blocks, args.hidden)
    train(dataset, config, args.output, args.epochs, args.batch_size, args.learning_rate, args.max_train, args.seed)


if __name__ == "__main__":
    main()