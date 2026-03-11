from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

Observation = dict[str, np.ndarray | float | int]


@dataclass(slots=True)
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class GameAdapter(ABC):
    @property
    @abstractmethod
    def action_space_n(self) -> int:
        """Number of discrete actions."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """Reset episode and return initial observation."""

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Apply one action and return transition data."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


class Trainer(ABC):
    @abstractmethod
    def train(self) -> Path:
        """Run training and return the latest checkpoint path."""

    @abstractmethod
    def evaluate(self, checkpoint: Path | None = None, episodes: int = 5) -> dict[str, float]:
        """Run evaluation and return scalar metrics."""


class Watcher(ABC):
    @abstractmethod
    def run(self) -> None:
        """Start a visible watcher session."""
