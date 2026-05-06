from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import UrbanSound8KDataset


def main() -> None:
    train_dataset = UrbanSound8KDataset(split="train")
    validation_dataset = UrbanSound8KDataset(split="validation")

    try:
        features, label = validation_dataset[0]
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None

    print(f"train examples: {len(train_dataset)}")
    print(f"validation examples: {len(validation_dataset)}")
    print(f"feature shape: {tuple(features.shape)}")
    print(f"label: {label}")


if __name__ == "__main__":
    main()
