from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .dataset import UrbanSound8KDataset
from .model import build_model_from_config
from .utils import ensure_dir, get_device, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained UrbanSound8K CNN checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/best_model.pt"),
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional validation batch limit for quick smoke tests.",
    )
    return parser.parse_args()


def create_validation_loader(config: dict[str, Any]) -> DataLoader:
    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))

    validation_dataset = UrbanSound8KDataset(config=config, split="validation")
    return DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint is missing model_state_dict: {path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[list[int], list[int]]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []

    for batch_index, (features, labels) in enumerate(dataloader, start=1):
        features = features.to(device)
        logits = model(features)
        predicted_labels = logits.argmax(dim=1)

        predictions.extend(predicted_labels.cpu().tolist())
        targets.extend(labels.tolist())

        if max_batches is not None and batch_index >= max_batches:
            break

    if not targets:
        raise ValueError("Validation dataloader produced no examples")

    return predictions, targets


def build_metrics(
    predictions: list[int],
    targets: list[int],
    class_id_to_name: dict[int, str],
) -> dict[str, Any]:
    class_ids = sorted(class_id_to_name)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=class_ids,
        zero_division=0,
    )

    per_class = {}
    for index, class_id in enumerate(class_ids):
        per_class[str(class_id)] = {
            "class_name": class_id_to_name[class_id],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    matrix = confusion_matrix(targets, predictions, labels=class_ids)

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, labels=class_ids, average="macro", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_ids": class_ids,
        "class_names": [class_id_to_name[class_id] for class_id in class_ids],
        "num_examples": len(targets),
    }


def save_confusion_matrix_plot(
    matrix: list[list[int]],
    class_names: list[str],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)

    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(values, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = values.max() / 2 if values.size and values.max() > 0 else 0
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            color = "white" if value > threshold else "black"
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    project_config = config.get("project", {})
    outputs_config = config.get("outputs", {})

    set_seed(int(project_config.get("seed", 42)))

    device = get_device()
    validation_loader = create_validation_loader(config)
    model = build_model_from_config(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)

    predictions, targets = collect_predictions(
        model=model,
        dataloader=validation_loader,
        device=device,
        max_batches=args.max_val_batches,
    )
    metrics = build_metrics(
        predictions=predictions,
        targets=targets,
        class_id_to_name=validation_loader.dataset.class_id_to_name,
    )

    metrics["checkpoint"] = str(args.checkpoint)
    metrics["checkpoint_epoch"] = checkpoint.get("epoch")
    metrics["checkpoint_val_macro_f1"] = checkpoint.get("val_macro_f1")

    metrics_dir = ensure_dir(outputs_config.get("metrics_dir", "outputs/metrics"))
    plots_dir = ensure_dir(outputs_config.get("plots_dir", "outputs/plots"))
    metrics_path = metrics_dir / "evaluation_metrics.json"
    matrix_path = plots_dir / "confusion_matrix.png"

    save_json(metrics, metrics_path)
    save_confusion_matrix_plot(
        matrix=metrics["confusion_matrix"],
        class_names=metrics["class_names"],
        output_path=matrix_path,
    )

    print(f"Using device: {device}")
    print(f"Validation examples evaluated: {metrics['num_examples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Confusion matrix plot: {matrix_path}")

    return metrics


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
