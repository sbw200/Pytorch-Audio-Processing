from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .dataset import UrbanSound8KDataset
from .model import build_model_from_config
from .utils import (
    ensure_dir,
    get_device,
    load_config,
    save_json,
    save_training_curves,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UrbanSound8K CNN classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional validation batch limit for quick smoke tests.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for quick smoke tests.",
    )
    return parser.parse_args()


def create_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))

    train_dataset = UrbanSound8KDataset(config=config, split="train")
    validation_dataset = UrbanSound8KDataset(config=config, split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch_index, (features, labels) in enumerate(dataloader, start=1):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if max_batches is not None and batch_index >= max_batches:
            break

    if total_examples == 0:
        raise ValueError("Training dataloader produced no examples")

    return total_loss / total_examples


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: list[int] = []
    targets: list[int] = []

    for batch_index, (features, labels) in enumerate(dataloader, start=1):
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)
        predicted_labels = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        predictions.extend(predicted_labels.cpu().tolist())
        targets.extend(labels.cpu().tolist())

        if max_batches is not None and batch_index >= max_batches:
            break

    if total_examples == 0:
        raise ValueError("Validation dataloader produced no examples")

    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    accuracy = correct / total_examples
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_macro_f1: float,
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    path = Path(checkpoint_path)
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_macro_f1": val_macro_f1,
            "config": config,
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, list[float]]:
    config = load_config(args.config)
    project_config = config.get("project", {})
    training_config = config.get("training", {})
    outputs_config = config.get("outputs", {})

    set_seed(int(project_config.get("seed", 42)))

    device = get_device()
    train_loader, validation_loader = create_dataloaders(config)
    model = build_model_from_config(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 0.001)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )

    epochs = int(args.epochs if args.epochs is not None else training_config.get("epochs", 10))
    best_macro_f1 = -1.0

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    checkpoints_dir = ensure_dir(outputs_config.get("checkpoints_dir", "outputs/checkpoints"))
    metrics_dir = ensure_dir(outputs_config.get("metrics_dir", "outputs/metrics"))
    plots_dir = ensure_dir(outputs_config.get("plots_dir", "outputs/plots"))

    checkpoint_path = checkpoints_dir / "best_model.pt"
    history_path = metrics_dir / "training_history.json"
    curves_path = plots_dir / "training_curves.png"

    print(f"Using device: {device}")
    print(f"Train examples: {len(train_loader.dataset)}")
    print(f"Validation examples: {len(validation_loader.dataset)}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
        )
        val_metrics = validate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_macro_f1=best_macro_f1,
                config=config,
                checkpoint_path=checkpoint_path,
            )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    save_json(history, history_path)
    save_training_curves(history, curves_path)

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"Training curves: {curves_path}")

    return history


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
