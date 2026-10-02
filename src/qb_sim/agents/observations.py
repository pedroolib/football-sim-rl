"""Structured, role-local observations shared by agents and future policies."""

from dataclasses import dataclass

from qb_sim.engine.entities import BallStatus, Role
from qb_sim.engine.geometry import Vector2


@dataclass(frozen=True, slots=True)
class PlayerView:
    """Position and velocity of another player, as seen by an agent."""

    player_id: str
    position: Vector2
    velocity: Vector2


@dataclass(frozen=True, slots=True)
class ReceiverObservation:
    """Local information available to a receiver."""

    player_id: str
    role: Role
    position: Vector2
    velocity: Vector2
    quarterback_position: Vector2
    ball_status: BallStatus
    ball_position: Vector2
    ball_height: float
    nearest_defender: PlayerView | None
    time_step: float
    elapsed_seconds: float
    carrier_id: str | None = None


@dataclass(frozen=True, slots=True)
class DefenderObservation:
    """Local information available to a defender."""

    player_id: str
    role: Role
    position: Vector2
    velocity: Vector2
    target_id: str | None
    target: PlayerView | None
    ball_status: BallStatus
    ball_position: Vector2
    ball_height: float
    carrier_id: str | None
    time_step: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class QuarterbackObservation:
    """Information available to the quarterback, including pressure."""

    player_id: str
    role: Role
    position: Vector2
    velocity: Vector2
    pressure: float
    ball_status: BallStatus
    receivers: tuple[PlayerView, ...]
    defenders: tuple[PlayerView, ...]
    time_step: float
    elapsed_seconds: float
    time_remaining: float


Observation = ReceiverObservation | DefenderObservation | QuarterbackObservation
