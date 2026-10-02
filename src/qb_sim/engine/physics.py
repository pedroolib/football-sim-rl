"""Fixed-step movement and swept collision helpers."""

import numpy as np

from qb_sim.engine.entities import BallState, BallStatus
from qb_sim.engine.geometry import FieldBounds, Vector2, segment_circle_contact_fraction, vec2


def move_toward(position: Vector2, target: Vector2, max_distance: float) -> Vector2:
    """Return a copy of ``position`` moved toward ``target`` without overshooting."""
    if max_distance < 0.0:
        raise ValueError("max_distance must be non-negative")

    current = np.array(position, dtype=np.float64, copy=True)
    offset = target - current
    remaining_distance = float(np.linalg.norm(offset))
    if remaining_distance <= max_distance or remaining_distance == 0.0:
        return np.array(target, dtype=np.float64, copy=True)
    if max_distance == 0.0:
        return current
    return current + offset * (max_distance / remaining_distance)


def moving_circle_contact_fraction(
    first_start: Vector2,
    first_end: Vector2,
    second_start: Vector2,
    second_end: Vector2,
    radius: float,
) -> float | None:
    """Return the first tick fraction two moving points come within ``radius``."""
    relative_start = first_start - second_start
    relative_end = first_end - second_end
    return segment_circle_contact_fraction(relative_start, relative_end, vec2(0.0, 0.0), radius)


def segment_field_exit_fraction(
    start: Vector2,
    end: Vector2,
    field: FieldBounds,
) -> float | None:
    """Return the first segment fraction at which a point exits the field."""
    if not field.contains(start):
        return 0.0
    if field.contains(end):
        return None

    candidates: list[float] = []
    for axis, upper_bound in ((0, field.length_yards), (1, field.width_yards)):
        delta = float(end[axis] - start[axis])
        if end[axis] < 0.0 and delta < 0.0:
            candidates.append(float((0.0 - start[axis]) / delta))
        elif end[axis] > upper_bound and delta > 0.0:
            candidates.append(float((upper_bound - start[axis]) / delta))

    return min(candidates) if candidates else None


def advance_ball(ball: BallState, dt: float) -> tuple[Vector2, float]:
    """Return the ball's tentative position and fraction of the tick until its target."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if ball.status is not BallStatus.IN_FLIGHT or ball.target is None:
        raise ValueError("only an in-flight ball with a target can advance")

    offset = ball.target - ball.position
    remaining_distance = float(np.linalg.norm(offset))
    speed = float(np.linalg.norm(ball.velocity))
    if remaining_distance == 0.0:
        return np.array(ball.target, dtype=np.float64, copy=True), 0.0
    if speed == 0.0:
        return np.array(ball.position, dtype=np.float64, copy=True), 1.0

    tick_distance = speed * dt
    if remaining_distance <= tick_distance:
        return (
            np.array(ball.target, dtype=np.float64, copy=True),
            remaining_distance / tick_distance,
        )

    return np.array(ball.position + ball.velocity * dt, dtype=np.float64), 1.0
