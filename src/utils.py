from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)

    epochs = history.get("epoch", [])
    if not epochs:
        raise ValueError("Cannot plot training curves with empty history")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_accuracy"], color="tab:green")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["val_macro_f1"], color="tab:orange")
    axes[2].set_title("Validation Macro F1")
    axes[2].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
