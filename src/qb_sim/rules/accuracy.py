"""Contextual throwing accuracy: launch-condition deviation, not outcome luck."""

from collections.abc import Sequence
from math import radians

from qb_sim.config import SimulationConfig
from qb_sim.engine.geometry import Vector2, distance


def pressure_from_defenders(
    quarterback_position: Vector2,
    defender_positions: Sequence[Vector2],
    radius: float,
) -> float:
    """Return 0..1 pressure from the nearest defender within ``radius``."""
    if radius <= 0.0 or not defender_positions:
        return 0.0
    nearest = min(distance(quarterback_position, position) for position in defender_positions)
    return _clamp(1.0 - nearest / radius)


def throw_spread(
    qb_speed: float,
    pressure: float,
    distance_yards: float,
    config: SimulationConfig,
) -> tuple[float, float]:
    """Return ``(angle_sigma_radians, power_sigma)`` for the throw context.

    Planted throws without pressure at short distance are the accuracy
    reference; running, pressure, and distance multiply the base spread.
    """
    speed_ratio = _clamp(qb_speed / config.quarterback_speed)
    distance_ratio = _clamp(distance_yards / config.throw_max_range)
    multiplier = (
        (1.0 + config.throw_move_penalty * speed_ratio)
        * (1.0 + config.throw_pressure_penalty * pressure)
        * (1.0 + config.throw_distance_penalty * distance_ratio)
    )
    return (
        radians(config.throw_base_sigma_degrees) * multiplier,
        config.throw_power_sigma * multiplier,
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
