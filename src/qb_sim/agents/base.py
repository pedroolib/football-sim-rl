"""Common agent interface shared by rule controllers and learned policies."""

from abc import ABC, abstractmethod

from qb_sim.agents.observations import Observation
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayState, Role


class BaseAgent(ABC):
    """One instance per player: observes local state and acts on its role."""

    role: Role

    def __init__(self, player_id: str) -> None:
        self.player_id = player_id

    def reset(self) -> None:  # noqa: B027 - optional hook, not all agents have state
        """Reset per-play state; controllers with routes override this."""

    @abstractmethod
    def observe(self, state: PlayState) -> Observation:
        """Build this role's local observation from the full play state."""

    @abstractmethod
    def act(self, observation: Observation) -> Move | Throw | None:
        """Choose an action allowed for this role, or ``None`` to stand still."""

    def decide(self, state: PlayState) -> Move | Throw | None:
        """Convenience composition used by the headless loop."""
        return self.act(self.observe(state))
