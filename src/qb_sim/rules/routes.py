"""Waypoint-based deterministic receiver routes."""

from dataclasses import dataclass
from enum import StrEnum
from math import cos, radians, sin

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.geometry import FieldBounds, Vector2, vec2


class RouteName(StrEnum):
    """Initial routes available to the receiver."""

    GO = "go"
    SLANT = "slant"
    POST = "post"


@dataclass(slots=True)
class Route:
    """A finite list of field-coordinate waypoints followed at a capped speed."""

    waypoints: tuple[Vector2, ...]
    _active_index: int = 0

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("a route must have at least one waypoint")
        self.waypoints = tuple(
            np.array(point, dtype=np.float64, copy=True) for point in self.waypoints
        )

    def advance(self, position: Vector2, max_distance: float) -> tuple[Vector2, bool]:
        """Move along waypoints by at most ``max_distance`` yards."""
        if max_distance < 0.0:
            raise ValueError("max_distance must be non-negative")

        moved = np.array(position, dtype=np.float64, copy=True)
        remaining = max_distance
        while self._active_index < len(self.waypoints):
            target = self.waypoints[self._active_index]
            offset = target - moved
            segment_length = float(np.linalg.norm(offset))
            if segment_length <= remaining:
                moved = target.copy()
                remaining -= segment_length
                self._active_index += 1
                continue
            if segment_length > 0.0 and remaining > 0.0:
                moved += offset * (remaining / segment_length)
            return moved, False

        return moved, True


def build_route(
    name: RouteName,
    start: Vector2,
    field: FieldBounds,
    config: SimulationConfig,
) -> Route:
    """Build a route whose breaks point inward from the receiver's side."""
    if not field.contains(start):
        raise ValueError("route start must be inside the field")

    start_point = np.array(start, dtype=np.float64, copy=True)
    field_center_y = field.width_yards / 2.0
    inward_sign = -1.0 if start_point[1] > field_center_y else 1.0

    if name is RouteName.GO:
        waypoints = (_clamp_to_field(start_point + vec2(config.go_length_yards, 0.0), field),)
    elif name is RouteName.SLANT:
        break_point = _clamp_to_field(start_point + vec2(config.slant_break_yards, 0.0), field)
        waypoints = (
            break_point,
            _cut_inward(
                break_point,
                inward_sign,
                config.slant_angle_degrees,
                config.cut_length_yards,
                field,
            ),
        )
    elif name is RouteName.POST:
        break_point = _clamp_to_field(start_point + vec2(config.post_break_yards, 0.0), field)
        waypoints = (
            break_point,
            _cut_inward(
                break_point,
                inward_sign,
                config.post_angle_degrees,
                config.cut_length_yards,
                field,
            ),
        )
    else:
        raise ValueError(f"unsupported route: {name!r}")

    return Route(waypoints)


def _cut_inward(
    start: Vector2,
    inward_sign: float,
    angle_degrees: float,
    length_yards: float,
    field: FieldBounds,
) -> Vector2:
    angle = radians(angle_degrees)
    offset = vec2(length_yards * cos(angle), inward_sign * length_yards * sin(angle))
    return _clamp_to_field(start + offset, field)


def _clamp_to_field(point: Vector2, field: FieldBounds) -> Vector2:
    return np.array(
        (
            min(max(float(point[0]), 0.0), field.length_yards),
            min(max(float(point[1]), 0.0), field.width_yards),
        ),
        dtype=np.float64,
    )
