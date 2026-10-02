"""Mutable state records for players, the ball, and a play."""

from dataclasses import dataclass
from enum import Enum

from qb_sim.geometry import Vector2


class BallStatus(Enum):
    """The ball's current simulation state."""

    HELD = "held"
    IN_FLIGHT = "in_flight"
    DEAD = "dead"


class PlayOutcome(Enum):
    """Possible terminal and non-terminal play outcomes."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INTERCEPTION = "interception"
    INCOMPLETE = "incomplete"
    OUT_OF_BOUNDS = "out_of_bounds"
    TIME_LIMIT = "time_limit"


@dataclass(slots=True)
class PlayerState:
    """A player's current position and velocity in yards and yards per second."""

    position: Vector2
    velocity: Vector2


@dataclass(slots=True)
class BallState:
    """Football position, velocity, target, and status."""

    position: Vector2
    velocity: Vector2
    status: BallStatus = BallStatus.HELD
    target: Vector2 | None = None


@dataclass(slots=True)
class PlayState:
    """Full mutable state needed to advance or inspect a single play."""

    quarterback: PlayerState
    receiver: PlayerState
    defender: PlayerState
    ball: BallState
    elapsed_seconds: float = 0.0
    outcome: PlayOutcome = PlayOutcome.IN_PROGRESS
