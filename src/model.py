from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Small Conv2D block for Mel spectrogram feature maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class AudioCNNClassifier(nn.Module):
    """Lightweight CNN classifier for Mel spectrogram inputs."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        conv_channels: Sequence[int] = (16, 32, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one channel size")
        if any(channel <= 0 for channel in conv_channels):
            raise ValueError(f"conv_channels must be positive, got {conv_channels}")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        channels = [input_channels, *conv_channels]
        self.features = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=channels[index],
                    out_channels=channels[index + 1],
                    dropout=dropout,
                )
                for index in range(len(conv_channels))
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(conv_channels[-1], num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                "Expected input with shape [batch, channels, n_mels, frames], "
                f"got {tuple(inputs.shape)}"
            )

        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)


def build_model_from_config(config: dict[str, Any]) -> AudioCNNClassifier:
    model_config = config.get("model", {})

    return AudioCNNClassifier(
        num_classes=int(model_config.get("num_classes", 10)),
        input_channels=int(model_config.get("input_channels", 1)),
        conv_channels=tuple(model_config.get("conv_channels", [16, 32, 64])),
        dropout=float(model_config.get("dropout", 0.3)),
    )
