"""Pure translation between the 6-value environment action and engine actions.

Action vector (every component in ``[-1, 1]``):
``[move_x, move_y, aim_x, aim_y, power_raw, throw_trigger]``.
A positive trigger always throws toward ``(aim_x, aim_y)`` (defaulting to
``(1, 0)`` when the aim is nearly zero) with ``power = (power_raw + 1) / 2``;
otherwise the QB moves in ``(move_x, move_y)``.
"""

import numpy as np

from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus
from qb_sim.engine.geometry import vec2

ACTION_SIZE = 6
THROW_TRIGGER_THRESHOLD = 0.0


def decode_action(
    action: np.ndarray,
    ball_status: BallStatus,
) -> dict[str, Move | Throw]:
    """Translate one environment action into engine actions for the QB."""
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (ACTION_SIZE,):
        raise ValueError(f"action must have {ACTION_SIZE} values")

    trigger = float(values[5])
    if trigger > THROW_TRIGGER_THRESHOLD and ball_status is BallStatus.HELD:
        aim = vec2(float(values[2]), float(values[3]))
        if float(np.linalg.norm(aim)) <= 1e-9:
            aim = vec2(1.0, 0.0)
        power = min(max((float(values[4]) + 1.0) / 2.0, 0.0), 1.0)
        return {"qb": Throw(direction=aim, power=power)}

    move = vec2(float(values[0]), float(values[1]))
    length = float(np.linalg.norm(move))
    if length <= 1e-9:
        return {}
    if length > 1.0:
        move = move / length
    return {"qb": Move(direction=move)}


def encode_engine_action(action: Move | Throw) -> np.ndarray:
    """Encode an engine action back into the environment action vector."""
    if isinstance(action, Throw):
        return np.array(
            [
                0.0,
                0.0,
                float(action.direction[0]),
                float(action.direction[1]),
                action.power * 2.0 - 1.0,
                1.0,
            ],
            dtype=np.float32,
        )
    if isinstance(action, Move):
        return np.array(
            [
                float(action.direction[0]),
                float(action.direction[1]),
                0.0,
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float32,
        )
    raise TypeError(f"unsupported action: {type(action).__name__}")
