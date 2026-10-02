"""Keyboard and mouse gesture translation into engine actions."""

from dataclasses import dataclass

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.geometry import Vector2, vec2

_MOVE_KEYS = {
    "w": vec2(0.0, 1.0),
    "up": vec2(0.0, 1.0),
    "s": vec2(0.0, -1.0),
    "down": vec2(0.0, -1.0),
    "a": vec2(-1.0, 0.0),
    "left": vec2(-1.0, 0.0),
    "d": vec2(1.0, 0.0),
    "right": vec2(1.0, 0.0),
}


def direction_from_keys(held: frozenset[str]) -> Vector2:
    """Combine movement keys into a unit direction, or zero when none apply."""
    direction = vec2(0.0, 0.0)
    for key in held:
        offset = _MOVE_KEYS.get(key)
        if offset is not None:
            direction = direction + offset
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return direction
    return direction / length


@dataclass(frozen=True, slots=True)
class Aim:
    """A validated drag aim: direction, landing point, and power fraction."""

    origin: Vector2
    direction: Vector2
    target: Vector2
    power: float


def aim_from_drag(
    quarterback_position: Vector2,
    drag: Vector2,
    config: SimulationConfig,
) -> Aim | None:
    """Convert a slingshot drag into an aim preview, or ``None`` when too short.

    Pulling back from the press point throws in the opposite direction, like a
    slingshot: the gesture length maps to power and range between the configured
    minimum and maximum. This is the same ``(angle, force)`` representation an
    RL policy can output.
    """
    drag_vector = np.asarray(drag, dtype=np.float64)
    length = float(np.linalg.norm(drag_vector))
    if length < config.throw_min_drag:
        return None

    direction = -drag_vector / length
    clamped_length = min(length, config.throw_max_drag)
    drag_span = config.throw_max_drag - config.throw_min_drag
    power = (clamped_length - config.throw_min_drag) / drag_span
    range_yards = config.throw_min_range + power * (config.throw_max_range - config.throw_min_range)
    target = quarterback_position + direction * range_yards
    return Aim(
        origin=np.array(quarterback_position, dtype=np.float64, copy=True),
        direction=direction,
        target=target,
        power=power,
    )
