"""Context-enriched role observations for Stage B (WR 28, defenders 24).

WR layout: 19 base + route one-hot (8) + yards to the goal line.
Defender layout: 19 base + down one-hot (3) + distance/40 + yards to the goal line.
"""

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayState
from qb_sim.rules.drive import DriveState
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER, RouteName
from qb_sim.training.role_observations import (
    defender_observation,
    receiver_observation,
)

RECEIVER_CONTEXT_OBSERVATION_SIZE = 19 + len(ROUTE_OBSERVATION_ORDER) + 1
DEFENDER_CONTEXT_OBSERVATION_SIZE = 19 + 3 + 2
DISTANCE_SCALE = 40.0


def receiver_context_observation(
    state: PlayState,
    player_id: str,
    config: SimulationConfig,
    route: RouteName | None,
    goal_line: float = 100.0,
) -> np.ndarray:
    """Receiver observation plus its called route and yards to the end zone."""
    values = [float(value) for value in receiver_observation(state, player_id, config)]
    values.extend(1.0 if route is route_name else 0.0 for route_name in ROUTE_OBSERVATION_ORDER)
    position_x = float(state.player(player_id).position[0])
    values.append((float(goal_line) - position_x) / float(goal_line))
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
