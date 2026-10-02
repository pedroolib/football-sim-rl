"""Fixed-size normalized observations per role for the multi-agent environment.

Receivers and defenders share a 19-float layout (role-local information only):

- 0-3: own position and velocity
- 4-7: quarterback relative position and velocity
- 8-11: ball status one-hot
- 12: ball height
- 13-16: target relative position and velocity (nearest defender for a
  receiver; assigned receiver, rusher target, or carrier for a defender)
- 17: target presence flag
- 18: remaining time fraction
"""

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayerState, PlayState, Role
from qb_sim.training.observations import BALL_STATUS_ORDER

ROLE_OBSERVATION_SIZE = 19


def receiver_observation(
    state: PlayState,
    player_id: str,
    config: SimulationConfig,
) -> np.ndarray:
    """Local observation for a receiver: self, QB, ball, nearest defender."""
    receiver = state.player(player_id)
    nearest: PlayerState | None = None
    nearest_distance = float("inf")
    for player in state.players:
        if player.role is not Role.DEFENDER:
            continue
        distance = float(np.linalg.norm(player.position - receiver.position))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = player
    return _role_observation(state, receiver, nearest, config)


def defender_observation(
    state: PlayState,
    player_id: str,
    target_id: str | None,
    config: SimulationConfig,
) -> np.ndarray:
    """Local observation for a defender: self, QB, ball, and its target."""
    defender = state.player(player_id)
    target = state.player(target_id) if target_id is not None else None
    return _role_observation(state, defender, target, config)


def _role_observation(
    state: PlayState,
    self_player: PlayerState,
    target: PlayerState | None,
    config: SimulationConfig,
) -> np.ndarray:
    field = config.field
    quarterback = state.player("qb")
    values: list[float] = [
        self_player.position[0] / field.length_yards,
        self_player.position[1] / field.width_yards,
        self_player.velocity[0] / _player_speed(self_player, config),
        self_player.velocity[1] / _player_speed(self_player, config),
        (quarterback.position[0] - self_player.position[0]) / field.length_yards,
        (quarterback.position[1] - self_player.position[1]) / field.width_yards,
        quarterback.velocity[0] / config.quarterback_speed,
        quarterback.velocity[1] / config.quarterback_speed,
    ]
    values.extend(1.0 if state.ball.status is status else 0.0 for status in BALL_STATUS_ORDER)
    values.append(state.ball.height / config.max_arc_height)
    if target is not None:
        values.extend(
            (
                (target.position[0] - self_player.position[0]) / field.length_yards,
                (target.position[1] - self_player.position[1]) / field.width_yards,
                target.velocity[0] / _player_speed(target, config),
                target.velocity[1] / _player_speed(target, config),
            )
        )
        values.append(1.0)
    else:
        values.extend((0.0, 0.0, 0.0, 0.0))
        values.append(0.0)
    values.append(1.0 - state.elapsed_seconds / config.max_play_duration)
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)


def _player_speed(player: PlayerState, config: SimulationConfig) -> float:
    if player.max_speed > 0.0:
        return player.max_speed
    if player.role is Role.QUARTERBACK:
        return config.quarterback_speed
    if player.role is Role.RECEIVER:
        return config.receiver_speed
    return config.defender_speed
