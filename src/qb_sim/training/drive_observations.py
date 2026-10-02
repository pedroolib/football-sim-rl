"""Fixed-size drive observation for the quarterback environment (v11).

Layout (57 floats in ``[-1, 1]``): the 36 passing-environment values first,
then route one-hots per receiver slot (8 routes each), then down one-hot (3),
distance to the marker normalized by 40, and yards to the goal line.
"""

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayState
from qb_sim.rules.drive import DriveState
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER, RouteName
from qb_sim.training.observations import OBSERVATION_SIZE, observation_from_state

RECEIVER_SLOTS = 2
ROUTE_SLOTS = len(ROUTE_OBSERVATION_ORDER)
DOWN_SLOTS = 3
DISTANCE_SCALE = 40.0
DRIVE_OBSERVATION_SIZE = OBSERVATION_SIZE + RECEIVER_SLOTS * ROUTE_SLOTS + DOWN_SLOTS + 2


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
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)
