"""Normalized observation builder shared by the environment and the viewer."""

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import BallStatus, PlayState, Role

_BALL_STATUS_ORDER = (
    BallStatus.HELD,
    BallStatus.IN_FLIGHT,
    BallStatus.CARRIED,
    BallStatus.DEAD,
)


def observation_size(config: SimulationConfig) -> int:
    """Number of floats in the observation for a given roster."""
    return (
        4
        + len(_BALL_STATUS_ORDER)
        + 1
        + 4 * len(config.receiver_starts)
        + 4 * (len(config.defender_starts) + len(config.rusher_starts))
        + 1
        + 1
    )


OBSERVATION_SIZE = observation_size(SimulationConfig())


def observation_from_state(state: PlayState, config: SimulationConfig) -> np.ndarray:
    """Build the 27-float normalized observation of the play from the QB's view."""
    field = config.field
    quarterback = state.player("qb")
    values: list[float] = [
        quarterback.position[0] / field.length_yards,
        quarterback.position[1] / field.width_yards,
        quarterback.velocity[0] / config.quarterback_speed,
        quarterback.velocity[1] / config.quarterback_speed,
    ]
    values.extend(1.0 if state.ball.status is status else 0.0 for status in _BALL_STATUS_ORDER)
    values.append(state.ball.height / config.max_arc_height)

    for role, speed in (
        (Role.RECEIVER, config.receiver_speed),
        (Role.DEFENDER, config.defender_speed),
    ):
        for player in state.players:
            if player.role is not role:
                continue
            values.extend(
                (
                    (player.position[0] - quarterback.position[0]) / field.length_yards,
                    (player.position[1] - quarterback.position[1]) / field.width_yards,
                    player.velocity[0] / speed,
                    player.velocity[1] / speed,
                )
            )

    values.append(state.pressure)
    values.append(1.0 - state.elapsed_seconds / config.max_play_duration)
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)
