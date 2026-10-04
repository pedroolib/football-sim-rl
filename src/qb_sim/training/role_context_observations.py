"""Context-enriched role observations for Stage B (WR 35, defenders 27).

WR layout: 19 base + route one-hot (8) + yards to the goal line + next-waypoint
delta (length and width) + capped distance to that waypoint + own speed, nearest
defender speed, QB throw range, and QB speed.
WR drive layout: the 35 WR context values + down one-hot (3), distance/40, and
yards from the line of scrimmage to the goal line.
Defender layout: 19 base + down one-hot (3) + distance/40 + yards to the goal
line + own speed, target speed, and the last throw distance.
"""

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayerState, PlayState, Role
from qb_sim.engine.geometry import Vector2, vec2
from qb_sim.rules.drive import DriveState
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER, RouteName, build_route
from qb_sim.training.role_observations import (
    defender_observation,
    receiver_observation,
)

RECEIVER_CAPABILITY_SLOTS = 4
DEFENDER_CAPABILITY_SLOTS = 3
RECEIVER_CONTEXT_OBSERVATION_SIZE = (
    19 + len(ROUTE_OBSERVATION_ORDER) + 1 + 3 + RECEIVER_CAPABILITY_SLOTS
)
RECEIVER_DRIVE_OBSERVATION_SIZE = RECEIVER_CONTEXT_OBSERVATION_SIZE + 5
DEFENDER_CONTEXT_OBSERVATION_SIZE = 19 + 3 + 2 + DEFENDER_CAPABILITY_SLOTS
DISTANCE_SCALE = 40.0
WAYPOINT_DISTANCE_SCALE = 10.0
LAST_THROW_DISTANCE_SCALE = 50.0


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


def _player_speed(player: PlayerState, config: SimulationConfig) -> float:
    if player.max_speed > 0.0:
        return player.max_speed
    if player.role is Role.QUARTERBACK:
        return config.quarterback_speed
    if player.role is Role.RECEIVER:
        return config.receiver_speed
    return config.defender_speed


def _nearest_defender_speed(
    state: PlayState, player_id: str, config: SimulationConfig
) -> float:
    player = state.player(player_id)
    nearest: PlayerState | None = None
    nearest_distance = float("inf")
    for candidate in state.players:
        if candidate.role is not Role.DEFENDER:
            continue
        distance = float(np.linalg.norm(candidate.position - player.position))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = candidate
    if nearest is None:
        return 0.0
    return _player_speed(nearest, config)


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
    values.extend(
        (
            _player_speed(state.player(player_id), config) / 10.0,
            _nearest_defender_speed(state, player_id, config) / 10.0,
            config.throw_max_range / 50.0,
            config.quarterback_speed / 10.0,
        )
    )
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)


def receiver_drive_observation(
    state: PlayState,
    player_id: str,
    config: SimulationConfig,
    route: RouteName | None,
    drive_state: DriveState,
    goal_line: float = 100.0,
    start: Vector2 | None = None,
) -> np.ndarray:
    """Receiver context observation plus down, distance, and field position."""
    values = [
        float(value)
        for value in receiver_context_observation(
            state, player_id, config, route, goal_line, start
        )
    ]
    down = int(drive_state.down)
    values.extend(1.0 if down == index else 0.0 for index in range(1, 4))
    values.append(min(float(drive_state.distance), DISTANCE_SCALE) / DISTANCE_SCALE)
    values.append((float(goal_line) - float(drive_state.los)) / float(goal_line))
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
    target = state.player(target_id) if target_id is not None else None
    values.extend(
        (
            _player_speed(state.player(player_id), config) / 10.0,
            (_player_speed(target, config) / 10.0) if target is not None else 0.0,
            (drive_state.last_throw_distance or 0.0) / LAST_THROW_DISTANCE_SCALE,
        )
    )
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)
