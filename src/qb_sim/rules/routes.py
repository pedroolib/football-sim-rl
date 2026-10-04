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
    OUT = "out"
    CURL = "curl"
    FLAT = "flat"
    WHEEL = "wheel"
    CORNER = "corner"


ROUTE_OBSERVATION_ORDER: tuple[RouteName, ...] = (
    RouteName.GO,
    RouteName.SLANT,
    RouteName.POST,
    RouteName.OUT,
    RouteName.CURL,
    RouteName.FLAT,
    RouteName.WHEEL,
    RouteName.CORNER,
)
DEFAULT_ROUTE_POOL = ROUTE_OBSERVATION_ORDER

RED_ZONE_YARDS = 20.0
THIRD_AND_LONG_POOL = (RouteName.GO, RouteName.POST, RouteName.CORNER, RouteName.WHEEL)
THIRD_AND_SHORT_POOL = (RouteName.SLANT, RouteName.FLAT, RouteName.OUT, RouteName.CURL)
LONG_DISTANCE_POOL = (
    RouteName.GO,
    RouteName.POST,
    RouteName.CORNER,
    RouteName.WHEEL,
    RouteName.OUT,
)
RED_ZONE_POOL = (RouteName.SLANT, RouteName.OUT, RouteName.CORNER, RouteName.FLAT, RouteName.CURL)


def route_pool_for(
    down: int,
    distance: float,
    los: float,
    goal_line: float = 100.0,
) -> tuple[RouteName, ...]:
    """Situational route pool for a down, distance to the marker, and field position."""
    if goal_line - los <= RED_ZONE_YARDS:
        return RED_ZONE_POOL
    if down == 3 and distance >= 8.0:
        return THIRD_AND_LONG_POOL
    if down == 3 and distance <= 3.0:
        return THIRD_AND_SHORT_POOL
    if distance >= 8.0:
        return LONG_DISTANCE_POOL
    return DEFAULT_ROUTE_POOL


def sample_routes(
    rng: np.random.Generator,
    count: int,
    pool: tuple[RouteName, ...] = DEFAULT_ROUTE_POOL,
) -> tuple[RouteName, ...]:
    """Draw ``count`` routes uniformly from ``pool`` using ``rng``."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if not pool:
        raise ValueError("pool must not be empty")
    indices = rng.integers(0, len(pool), size=count)
    return tuple(pool[int(index)] for index in indices)


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

    @property
    def remaining_waypoints(self) -> tuple[Vector2, ...]:
        """Waypoints not yet reached, for read-only visualization."""
        return self.waypoints[self._active_index :]

    def reset(self) -> None:
        """Restart the route from its first waypoint."""
        self._active_index = 0

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
    max_depth_yards: float | None = None,
) -> Route:
    """Build a route whose breaks point inward from the receiver's side.

    When ``max_depth_yards`` is given, every waypoint's x is capped at
    ``start_x + max_depth_yards`` so receivers stay within the quarterback's arm.
    """
    if not field.contains(start):
        raise ValueError("route start must be inside the field")

    start_point = np.array(start, dtype=np.float64, copy=True)
    field_center_y = field.width_yards / 2.0
    inward_sign = -1.0 if start_point[1] > field_center_y else 1.0
    outward_sign = -inward_sign

    if name is RouteName.GO:
        waypoints = (_clamp_to_field(start_point + vec2(config.go_length_yards, 0.0), field),)
    elif name is RouteName.SLANT:
        break_point = _clamp_to_field(start_point + vec2(config.slant_break_yards, 0.0), field)
        waypoints = (
            break_point,
            _angled_cut(
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
            _angled_cut(
                break_point,
                inward_sign,
                config.post_angle_degrees,
                config.cut_length_yards,
                field,
            ),
        )
    elif name is RouteName.OUT:
        break_point = _clamp_to_field(start_point + vec2(config.out_break_yards, 0.0), field)
        waypoints = (
            break_point,
            _angled_cut(
                break_point,
                outward_sign,
                config.out_angle_degrees,
                config.cut_length_yards,
                field,
            ),
        )
    elif name is RouteName.CURL:
        waypoints = (
            _clamp_to_field(start_point + vec2(config.curl_depth_yards, 0.0), field),
            _clamp_to_field(
                start_point
                + vec2(config.curl_depth_yards - config.curl_back_yards, 0.0),
                field,
            ),
        )
    elif name is RouteName.FLAT:
        waypoints = (
            _clamp_to_field(
                start_point + vec2(0.0, outward_sign * config.flat_length_yards),
                field,
            ),
        )
    elif name is RouteName.WHEEL:
        break_point = _clamp_to_field(
            start_point + vec2(0.0, outward_sign * config.wheel_break_yards),
            field,
        )
        waypoints = (
            break_point,
            _clamp_to_field(break_point + vec2(config.wheel_length_yards, 0.0), field),
        )
    elif name is RouteName.CORNER:
        break_point = _clamp_to_field(start_point + vec2(config.corner_break_yards, 0.0), field)
        waypoints = (
            break_point,
            _angled_cut(
                break_point,
                outward_sign,
                config.corner_angle_degrees,
                config.cut_length_yards,
                field,
            ),
        )
    else:
        raise ValueError(f"unsupported route: {name!r}")

    if max_depth_yards is not None:
        depth_limit = float(start_point[0]) + max_depth_yards
        waypoints = tuple(
            np.array((min(float(point[0]), depth_limit), float(point[1])), dtype=np.float64)
            for point in waypoints
        )

    return Route(waypoints)


def _angled_cut(
    start: Vector2,
    sign: float,
    angle_degrees: float,
    length_yards: float,
    field: FieldBounds,
) -> Vector2:
    angle = radians(angle_degrees)
    offset = vec2(length_yards * cos(angle), sign * length_yards * sin(angle))
    return _clamp_to_field(start + offset, field)


def _clamp_to_field(point: Vector2, field: FieldBounds) -> Vector2:
    return np.array(
        (
            min(max(float(point[0]), 0.0), field.length_yards),
            min(max(float(point[1]), 0.0), field.width_yards),
        ),
        dtype=np.float64,
    )
