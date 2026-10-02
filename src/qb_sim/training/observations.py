"""Normalized observation builder shared by the environment and the viewer.

The layout is fixed across scenarios so one policy can play any roster:
slots that are empty in a scenario are zeroed and marked by a presence flag.

Layout (36 floats in ``[-1, 1]``):

- 0-3: quarterback position and velocity
- 4-7: ball status one-hot (held, in flight, carried, dead)
- 8: ball height
- 9-12 / 13-16: receiver slots (relative position, velocity)
- 17-18: receiver presence flags
- 19-22 / 23-26 / 27-30: defender slots, rushers included
- 31-33: defender presence flags
- 34: pressure
- 35: remaining time fraction
"""

import numpy as np

from qb_sim.config import MAX_DEFENDERS, MAX_RECEIVERS, SimulationConfig
from qb_sim.engine.entities import BallStatus, PlayerState, PlayState, Role

RECEIVER_SLOTS = MAX_RECEIVERS
DEFENDER_SLOTS = MAX_DEFENDERS
OBSERVATION_SIZE = 36
BALL_STATUS_ORDER = (
    BallStatus.HELD,
    BallStatus.IN_FLIGHT,
    BallStatus.CARRIED,
    BallStatus.DEAD,
)


def observation_from_state(state: PlayState, config: SimulationConfig) -> np.ndarray:
    """Build the fixed-size normalized observation from the QB's perspective."""
    field = config.field
    quarterback = state.player("qb")
    values: list[float] = [
        quarterback.position[0] / field.length_yards,
        quarterback.position[1] / field.width_yards,
        quarterback.velocity[0] / config.quarterback_speed,
        quarterback.velocity[1] / config.quarterback_speed,
    ]
    values.extend(1.0 if state.ball.status is status else 0.0 for status in BALL_STATUS_ORDER)
    values.append(state.ball.height / config.max_arc_height)

    receivers = [player for player in state.players if player.role is Role.RECEIVER]
    _extend_slots(values, quarterback, receivers, RECEIVER_SLOTS, config)
    for index in range(RECEIVER_SLOTS):
        values.append(1.0 if index < len(receivers) else 0.0)

    defenders = [player for player in state.players if player.role is Role.DEFENDER]
    _extend_slots(values, quarterback, defenders, DEFENDER_SLOTS, config)
    for index in range(DEFENDER_SLOTS):
        values.append(1.0 if index < len(defenders) else 0.0)

    values.append(state.pressure)
    values.append(1.0 - state.elapsed_seconds / config.max_play_duration)
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)


def _extend_slots(
    values: list[float],
    quarterback: PlayerState,
    players: list[PlayerState],
    slots: int,
    config: SimulationConfig,
) -> None:
    field = config.field
    for index in range(slots):
        if index < len(players):
            player = players[index]
            if player.max_speed > 0.0:
                speed = player.max_speed
            else:
                speed = (
                    config.receiver_speed if player.role is Role.RECEIVER else config.defender_speed
                )
            values.extend(
                (
                    (player.position[0] - quarterback.position[0]) / field.length_yards,
                    (player.position[1] - quarterback.position[1]) / field.width_yards,
                    player.velocity[0] / speed,
                    player.velocity[1] / speed,
                )
            )
        else:
            values.extend((0.0, 0.0, 0.0, 0.0))
