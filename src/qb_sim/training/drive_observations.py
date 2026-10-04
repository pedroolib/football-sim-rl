"""Fixed-size drive observation for the quarterback environment (v11).

Layout (62 floats in ``[-1, 1]``): the 36 passing-environment values first,
then route one-hots per receiver slot (8 routes each), then down one-hot (3),
distance to the marker normalized by 40, yards to the goal line, the QB
capability block (own speed, throw range, throw base sigma, best receiver
speed advantage), and the QB offset from the line of scrimmage (negative
behind it, positive past it).
"""

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayerState, PlayState, Role
from qb_sim.rules.drive import DriveState
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER, RouteName
from qb_sim.training.observations import OBSERVATION_SIZE, observation_from_state

RECEIVER_SLOTS = 2
ROUTE_SLOTS = len(ROUTE_OBSERVATION_ORDER)
DOWN_SLOTS = 3
CAPABILITY_SLOTS = 4
DISTANCE_SCALE = 40.0
DRIVE_OBSERVATION_SIZE = (
    OBSERVATION_SIZE
    + RECEIVER_SLOTS * ROUTE_SLOTS
    + DOWN_SLOTS
    + 2
    + CAPABILITY_SLOTS
    + 1
)


def drive_observation_from_state(
    state: PlayState,
    routes: tuple[RouteName, ...],
    drive_state: DriveState,
    config: SimulationConfig,
    goal_line: float = 100.0,
) -> np.ndarray:
    """Extend the QB observation with route, down, and field-position context."""
    values = [float(value) for value in observation_from_state(state, config)]
    for slot in range(RECEIVER_SLOTS):
        route = routes[slot] if slot < len(routes) else None
        values.extend(
            1.0 if route is route_name else 0.0 for route_name in ROUTE_OBSERVATION_ORDER
        )
    down = int(drive_state.down)
    values.extend(1.0 if down == index else 0.0 for index in range(1, DOWN_SLOTS + 1))
    values.append(min(float(drive_state.distance), DISTANCE_SCALE) / DISTANCE_SCALE)
    values.append((float(goal_line) - float(drive_state.los)) / float(goal_line))
    values.extend(_capability_features(state, config))
    values.append(
        (float(state.player("qb").position[0]) - float(drive_state.los))
        / config.field.length_yards
    )
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)


def _capability_features(state: PlayState, config: SimulationConfig) -> tuple[float, ...]:
    """QB capabilities plus the best receiver speed advantage over its cover."""
    quarterback = state.player("qb")
    receivers = [player for player in state.players if player.role is Role.RECEIVER]
    defenders = [player for player in state.players if player.role is Role.DEFENDER]
    advantage = 0.0
    for receiver in receivers:
        if not defenders:
            break
        nearest = min(
            defenders,
            key=lambda defender: float(
                np.linalg.norm(defender.position - receiver.position)
            ),
        )
        advantage = max(
            advantage,
            _player_speed(receiver, config) - _player_speed(nearest, config),
        )
    return (
        _player_speed(quarterback, config) / 10.0,
        config.throw_max_range / 50.0,
        config.throw_base_sigma_degrees / 5.0,
        max(0.0, advantage) / 2.0,
    )


def _player_speed(player: PlayerState, config: SimulationConfig) -> float:
    if player.max_speed > 0.0:
        return player.max_speed
    if player.role is Role.QUARTERBACK:
        return config.quarterback_speed
    if player.role is Role.RECEIVER:
        return config.receiver_speed
    return config.defender_speed
