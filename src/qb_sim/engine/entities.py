"""Mutable state records for players, the ball, and a play."""

from dataclasses import dataclass
from enum import Enum

from qb_sim.engine.geometry import Vector2


class BallStatus(Enum):
    """The ball's current simulation state."""

    HELD = "held"
    IN_FLIGHT = "in_flight"
    CARRIED = "carried"
    DEAD = "dead"


class PlayOutcome(Enum):
    """Possible terminal and non-terminal play outcomes."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INTERCEPTION = "interception"
    INCOMPLETE = "incomplete"
    OUT_OF_BOUNDS = "out_of_bounds"
    TIME_LIMIT = "time_limit"
    SACK = "sack"


class Role(Enum):
    """The football role a player state belongs to."""

    QUARTERBACK = "quarterback"
    RECEIVER = "receiver"
    DEFENDER = "defender"


@dataclass(slots=True)
class PlayerState:
    """A player's current position, velocity, role, and stable identifier."""

    position: Vector2
    velocity: Vector2
    role: Role | None = None
    player_id: str = ""
    max_speed: float = 0.0


@dataclass(slots=True)
class BallState:
    """Football position, velocity, target, status, and flight height."""

    position: Vector2
    velocity: Vector2
    status: BallStatus = BallStatus.HELD
    target: Vector2 | None = None
    height: float = 0.0
    launch_position: Vector2 | None = None
    flight_distance: float = 0.0
    arc_peak: float = 0.0


@dataclass(slots=True)
class PlayState:
    """Full mutable state needed to advance or inspect a single play."""

    players: list[PlayerState]
    ball: BallState
    elapsed_seconds: float = 0.0
    outcome: PlayOutcome = PlayOutcome.IN_PROGRESS
    carrier_id: str | None = None
    pressure: float = 0.0
    throw_time: float | None = None

    def player(self, player_id: str) -> PlayerState:
        """Return the player with ``player_id`` or raise ``KeyError``."""
        for player in self.players:
            if player.player_id == player_id:
                return player
        raise KeyError(player_id)
