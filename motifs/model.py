from dataclasses import dataclass

import torch
from torch import nn

from motifs.encoding import PLANES

REDUCED_CHANNELS = 8


@dataclass(frozen=True)
class ModelConfig:
    themes: int
    channels: int = 64
    blocks: int = 3
    hidden: int = 128


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.first_norm = nn.BatchNorm2d(channels)
        self.second = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.second_norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.first_norm(self.first(x)))
        out = self.second_norm(self.second(out))
        return torch.relu(out + x)


class MotifNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, config.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(config.channels) for _ in range(config.blocks)])
        self.reduce = nn.Sequential(
            nn.Conv2d(config.channels, REDUCED_CHANNELS, 1, bias=False),
            nn.BatchNorm2d(REDUCED_CHANNELS),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(REDUCED_CHANNELS * 64, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.themes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.reduce(self.blocks(self.stem(x))))


def build_model(config: ModelConfig) -> MotifNet:
    return MotifNet(config)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())