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

    players: list[PlayerState]
    ball: BallState
    elapsed_seconds: float = 0.0
    outcome: PlayOutcome = PlayOutcome.IN_PROGRESS
    carrier_id: str | None = None

    def player(self, player_id: str) -> PlayerState:
        """Return the player with ``player_id`` or raise ``KeyError``."""
        for player in self.players:
            if player.player_id == player_id:
                return player
        raise KeyError(player_id)
