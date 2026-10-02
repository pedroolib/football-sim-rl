"""Deterministic, headless quarterback passing simulation."""

from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import (
    BallState,
    BallStatus,
    PlayerState,
    PlayOutcome,
    PlayState,
    Role,
)
from qb_sim.engine.geometry import FieldBounds, Vector2, distance, vec2
from qb_sim.rules.play import Play
from qb_sim.rules.routes import Route, RouteName, build_route

__all__ = [
    "BallState",
    "BallStatus",
    "FieldBounds",
    "Move",
    "Play",
    "PlayOutcome",
    "PlayState",
    "PlayerState",
    "Role",
    "Route",
    "RouteName",
    "SimulationConfig",
    "Throw",
    "Vector2",
    "build_route",
    "distance",
    "vec2",
]
