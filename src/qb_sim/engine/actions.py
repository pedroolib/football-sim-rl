"""Hybrid action schema shared by humans, rule controllers, and RL policies."""

from dataclasses import dataclass

import numpy as np

from qb_sim.engine.entities import Role
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
    """Throw in a unit direction with a power fraction in ``[0, 1]``.

    Humans produce this action from a drag gesture and RL policies can learn
    it directly as an angle plus a force percentage.
    """

    direction: Vector2
    power: float = 1.0

    def __post_init__(self) -> None:
        direction = _validated_point(self.direction, "direction")
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            raise ValueError("direction must be non-zero")
        power = float(self.power)
        if not bool(np.isfinite(power)) or not 0.0 <= power <= 1.0:
            raise ValueError("power must be between 0 and 1")
        object.__setattr__(self, "direction", direction / length)
        object.__setattr__(self, "power", power)


def allowed_actions(role: Role) -> tuple[type, ...]:
    """Return the action types a player with ``role`` may execute."""
    if role is Role.QUARTERBACK:
        return (Move, Throw)
    if role in (Role.RECEIVER, Role.DEFENDER):
        return (Move,)
    raise ValueError(f"unsupported role: {role!r}")
