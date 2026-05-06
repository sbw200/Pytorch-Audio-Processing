from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model_from_config
from src.preprocessing import preprocess_audio_file
from src.utils import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one audio file through the trained UrbanSound8K classifier."
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
        required=True,
        help="Path to a trained model checkpoint.",
    )
    parser.add_argument(
        "--audio_path",
        type=Path,
        required=True,
        help="Path to a WAV file to classify.",
    )
    return parser.parse_args()


def load_class_names(config: dict[str, Any]) -> dict[int, str]:
    metadata_path = Path(config.get("data", {}).get("metadata_csv", ""))
    if metadata_path.exists():
        import pandas as pd

        metadata = pd.read_csv(metadata_path)
        if {"classID", "class"}.issubset(metadata.columns):
            classes = metadata[["classID", "class"]].drop_duplicates().sort_values("classID")
            return {int(row.classID): str(row["class"]) for _, row in classes.iterrows()}

    num_classes = int(config.get("model", {}).get("num_classes", 10))
    return {class_id: f"class_{class_id}" for class_id in range(num_classes)}


def validate_inputs(checkpoint_path: Path, audio_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint path is not a file: {checkpoint_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio path is not a file: {audio_path}")


def run_inference(
    config: dict[str, Any],
    checkpoint_path: Path,
    audio_path: Path,
) -> tuple[str, float, float]:
    device = get_device()
    class_names = load_class_names(config)

    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features = preprocess_audio_file(audio_path, config).unsqueeze(0).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class_id = probabilities.max(dim=1)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    class_id = int(predicted_class_id.item())
    class_name = class_names.get(class_id, f"class_{class_id}")
    return class_name, float(confidence.item()), elapsed_ms


def main() -> None:
    args = parse_args()

    try:
        validate_inputs(args.checkpoint, args.audio_path)
        config = load_config(args.config)
        predicted_class, confidence, elapsed_ms = run_inference(
            config=config,
            checkpoint_path=args.checkpoint,
            audio_path=args.audio_path,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None

    print(f"predicted_class: {predicted_class}")
    print(f"confidence: {confidence:.4f}")
    print(f"inference_time_ms: {elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
