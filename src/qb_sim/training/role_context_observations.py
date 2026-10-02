"""Context-enriched role observations for Stage B (WR 31, defenders 24).

WR layout: 19 base + route one-hot (8) + yards to the goal line + next-waypoint
delta (length and width) + capped distance to that waypoint.
Defender layout: 19 base + down one-hot (3) + distance/40 + yards to the goal line.
"""

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayState
from qb_sim.engine.geometry import Vector2, vec2
from qb_sim.rules.drive import DriveState
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER, RouteName, build_route
from qb_sim.training.role_observations import (
    defender_observation,
    receiver_observation,
)

RECEIVER_CONTEXT_OBSERVATION_SIZE = 19 + len(ROUTE_OBSERVATION_ORDER) + 1 + 3
DEFENDER_CONTEXT_OBSERVATION_SIZE = 19 + 3 + 2
DISTANCE_SCALE = 40.0
WAYPOINT_DISTANCE_SCALE = 10.0


def _next_route_waypoint(
    route: RouteName | None,
    start: Vector2,
    position: Vector2,
    config: SimulationConfig,
) -> Vector2 | None:
    if route is None:
        return None
    built = build_route(route, start, config.field, config)
    previous = start
    for waypoint in built.waypoints:
        segment = waypoint - previous
        denominator = float(segment.dot(segment))
        progress = (
            0.0
            if denominator == 0.0
            else float((position - previous).dot(segment) / denominator)
        )
        if progress < 0.999:
            return waypoint
        previous = waypoint
    return built.waypoints[-1]


def receiver_context_observation(
    state: PlayState,
    player_id: str,
    config: SimulationConfig,
    route: RouteName | None,
    goal_line: float = 100.0,
    start: Vector2 | None = None,
) -> np.ndarray:
    """Receiver observation plus its route, field position, and route progress.

    ``start`` overrides the unshifted formation start so callers playing a
    shifted line of scrimmage build route-progress features from the actual
    spot; ``None`` keeps the default formation derivation. While this player
    carries the ball the three route-progress features are zeroed: the old
    route waypoint is no longer a useful target and would steer the carrier
    back toward it instead of the end zone.
    """
    values = [float(value) for value in receiver_observation(state, player_id, config)]
    values.extend(1.0 if route is route_name else 0.0 for route_name in ROUTE_OBSERVATION_ORDER)
    position_x = float(state.player(player_id).position[0])
    values.append((float(goal_line) - position_x) / float(goal_line))
    index = int(player_id.removeprefix("wr")) - 1
    if start is None:
        start = vec2(*config.receiver_starts[index])
    if state.carrier_id == player_id:
        values.extend((0.0, 0.0, 0.0))
    else:
        waypoint = _next_route_waypoint(route, start, state.player(player_id).position, config)
        if waypoint is None:
            values.extend((0.0, 0.0, 0.0))
        else:
            delta = waypoint - state.player(player_id).position
            values.extend(
                (
                    float(delta[0]) / config.field.length_yards,
                    float(delta[1]) / config.field.width_yards,
                    min(float(np.linalg.norm(delta)), WAYPOINT_DISTANCE_SCALE)
                    / WAYPOINT_DISTANCE_SCALE,
                )
            )
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)


def defender_context_observation(
    state: PlayState,
    player_id: str,
    target_id: str | None,
    config: SimulationConfig,
    drive_state: DriveState,
    goal_line: float = 100.0,
) -> np.ndarray:
    """Defender observation plus down, distance, and field position."""
    values = [
        float(value)
        for value in defender_observation(state, player_id, target_id, config)
    ]
    down = int(drive_state.down)
    values.extend(1.0 if down == index else 0.0 for index in range(1, 4))
    values.append(min(float(drive_state.distance), DISTANCE_SCALE) / DISTANCE_SCALE)
    values.append((float(goal_line) - float(drive_state.los)) / float(goal_line))
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)
