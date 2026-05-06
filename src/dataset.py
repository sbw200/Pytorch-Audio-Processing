from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

try:
    from .preprocessing import audio_file_supported, preprocess_audio_file
except ImportError:
    from preprocessing import audio_file_supported, preprocess_audio_file


Split = Literal["train", "validation", "val", "all"]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return config


class UrbanSound8KDataset(Dataset):
    """UrbanSound8K dataset using the official fold assignments."""

    required_columns = {"slice_file_name", "fold", "classID", "class"}

    def __init__(
        self,
        root_dir: str | Path | None = None,
        metadata_csv: str | Path | None = None,
        config: dict[str, Any] | None = None,
        config_path: str | Path = "configs/config.yaml",
        split: Split = "train",
        validation_fold: int | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.config = config if config is not None else load_config(config_path)
        data_config = self.config.get("data", {})

        self.root_dir = Path(root_dir or data_config.get("dataset_root", "archive (4)"))
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"UrbanSound8K dataset directory not found: {self.root_dir}. "
                "Set data.dataset_root in configs/config.yaml or pass root_dir."
            )
        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"UrbanSound8K dataset path is not a directory: {self.root_dir}"
            )

        self.metadata_csv = Path(
            metadata_csv or data_config.get("metadata_csv", self.root_dir / "UrbanSound8K.csv")
        )
        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"UrbanSound8K metadata CSV not found: {self.metadata_csv}. "
                "Expected UrbanSound8K.csv in the dataset root or set data.metadata_csv."
            )

        self.split = split
        self.validation_fold = int(
            validation_fold
            if validation_fold is not None
            else data_config.get("validation_fold", 1)
        )
        if self.validation_fold < 1 or self.validation_fold > 10:
            raise ValueError(
                f"validation_fold must be between 1 and 10, got {self.validation_fold}"
            )

        self.return_metadata = return_metadata
        self.skip_unsupported_audio = bool(data_config.get("skip_unsupported_audio", True))
        self.metadata = self._load_metadata()
        self.metadata = self._apply_split(self.metadata)
        if self.skip_unsupported_audio:
            self.metadata = self._filter_supported_audio(self.metadata)

        if self.metadata.empty:
            raise ValueError(
                f"No UrbanSound8K rows found for split='{self.split}' "
                f"with validation_fold={self.validation_fold}"
            )

        classes = (
            self.metadata[["classID", "class"]]
            .drop_duplicates()
            .sort_values("classID")
        )
        self.class_id_to_name = {
            int(row.classID): str(row["class"]) for _, row in classes.iterrows()
        }

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, dict[str, Any]]:
        row = self.metadata.iloc[index]
        audio_path = self.root_dir / f"fold{int(row.fold)}" / str(row.slice_file_name)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file listed in metadata is missing: {audio_path}"
            )

        features = preprocess_audio_file(audio_path, self.config)
        label = int(row.classID)

        if self.return_metadata:
            return features, label, row.to_dict()
        return features, label

    def _load_metadata(self) -> pd.DataFrame:
        metadata = pd.read_csv(self.metadata_csv)
        missing_columns = self.required_columns.difference(metadata.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"UrbanSound8K metadata is missing required columns: {missing}"
            )

        return metadata.copy()

    def _apply_split(self, metadata: pd.DataFrame) -> pd.DataFrame:
        split = self.split.lower()
        if split == "train":
            return metadata[metadata["fold"] != self.validation_fold].reset_index(drop=True)
        if split in {"validation", "val"}:
            return metadata[metadata["fold"] == self.validation_fold].reset_index(drop=True)
        if split == "all":
            return metadata.reset_index(drop=True)

        raise ValueError(
            "split must be one of 'train', 'validation', 'val', or 'all', "
            f"got '{self.split}'"
        )

    def _filter_supported_audio(self, metadata: pd.DataFrame) -> pd.DataFrame:
        supported_rows = []
        skipped = 0

        for row in metadata.itertuples(index=False):
            audio_path = self.root_dir / f"fold{int(row.fold)}" / str(row.slice_file_name)
            try:
                supported = audio_file_supported(audio_path)
            except RuntimeError:
                supported = False

            if supported:
                supported_rows.append(row)
            else:
                skipped += 1

        if skipped:
            print(
                f"Skipped {skipped} unsupported compressed audio file(s). "
                "Set data.skip_unsupported_audio=false to fail instead."
            )

        if not supported_rows:
            return metadata.iloc[0:0].copy()

        return pd.DataFrame(supported_rows, columns=metadata.columns).reset_index(drop=True)
