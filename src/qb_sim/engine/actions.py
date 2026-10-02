"""Hybrid action schema shared by humans, rule controllers, and RL policies."""

from dataclasses import dataclass

import numpy as np

from qb_sim.engine.geometry import Vector2


def _validated_point(vector: Vector2, name: str) -> Vector2:
    point = np.asarray(vector, dtype=np.float64)
    if point.shape != (2,) or not bool(np.all(np.isfinite(point))):
        raise ValueError(f"{name} must be a finite two-dimensional point")
    return np.array(point, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class Move:
    """Move a controlled player in a direction of magnitude at most 1."""

    direction: Vector2

    def __post_init__(self) -> None:
        direction = _validated_point(self.direction, "direction")
        if float(np.linalg.norm(direction)) > 1.0 + 1e-9:
            raise ValueError("direction magnitude must be at most 1")
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True, slots=True)
class Throw:
    """Throw the ball toward a target point in field coordinates."""

    target: Vector2

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _validated_point(self.target, "target"))
