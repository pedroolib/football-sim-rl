"""Monitoring helpers for live training dashboards (no pygame, no torch)."""

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from qb_sim.agents.loop import build_rule_agents
from qb_sim.agents.rule_controllers import HeuristicQuarterback, ReceiverRouteController
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayOutcome, PlayState
from qb_sim.rules.play import Play
from qb_sim.rules.routes import Route, RouteName

QuarterbackPolicy = Callable[[PlayState], Move | Throw | None]

_CHECKPOINT_SUFFIX = "_steps.zip"
_HOLD_TICKS = 10


def checkpoint_steps(path: str | Path) -> int:
    """Extract the training step count from a checkpoint filename, or -1."""
    for part in reversed(Path(path).stem.split("_")):
        if part.isdigit():
            return int(part)
    return -1


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the newest checkpoint by training steps, or ``None``."""
    directory = Path(run_dir) / "checkpoints"
    if not directory.is_dir():
        return None
    checkpoints = list(directory.glob(f"*{_CHECKPOINT_SUFFIX}"))
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_steps)


def read_progress(run_dir: str | Path) -> list[tuple[int, float]]:
    """Read ``(timesteps, ep_rew_mean)`` pairs from the SB3 progress CSV."""
    path = Path(run_dir) / "logs" / "progress.csv"
    if not path.is_file():
        return []
    points: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timesteps = row.get("time/total_timesteps")
            reward = row.get("rollout/ep_rew_mean")
            if not timesteps or not reward:
                continue
            try:
                points.append((int(float(timesteps)), float(reward)))
            except ValueError:
                continue
    return points


def rule_quarterback_policy(config: SimulationConfig) -> QuarterbackPolicy:
    """A demo policy using the M3 heuristic quarterback."""
    agent = HeuristicQuarterback("qb", config)

    def policy(state: PlayState) -> Move | Throw | None:
        return agent.decide(state)

    return policy


@dataclass(slots=True)
class DemoPanel:
    """One live demo play with rule teammates and a swappable QB policy."""

    play: Play
    routes: tuple[RouteName, ...] = (RouteName.POST, RouteName.SLANT)
    policy: QuarterbackPolicy | None = None
    seed: int = 0
    completed: int = 0
    finished: int = 0
    last_outcome: str = ""
    _hold: int = field(default=0, repr=False)
    _agents: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._build_agents()

    def _build_agents(self) -> None:
        self._agents = build_rule_agents(
            self.play,
            self.routes,
            self.play.config,
            include_quarterback=False,
        )

    @property
    def receiver_routes(self) -> dict[str, Route]:
        """Routes of the rule receivers, for visualization."""
        return {
            player_id: agent.route
            for player_id, agent in self._agents.items()
            if isinstance(agent, ReceiverRouteController)
        }

    @property
    def completion_pct(self) -> float:
        """Completed plays as a percentage of finished plays."""
        return 100.0 * self.completed / self.finished if self.finished else 0.0

    def set_policy(self, policy: QuarterbackPolicy | None) -> None:
        """Swap the quarterback policy without resetting the play."""
        self.policy = policy

    def restart(self, seed: int | None = None) -> PlayState:
        """Start a new play with the next seed."""
        if seed is not None:
            self.seed = seed
        self.play.reset(seed=self.seed)
        self.seed += 1
        self._hold = 0
        self._build_agents()
        return self.play.state

    def tick(self) -> PlayState:
        """Advance one simulation tick; auto-restart after a short hold."""
        state = self.play.state
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            if self._hold > 0:
                self._hold -= 1
                return state
            return self.restart()

        actions: dict[str, Move | Throw] = {}
        for player_id, agent in self._agents.items():
            action = agent.decide(state)
            if action is not None:
                actions[player_id] = action
        if self.policy is not None:
            quarterback_action = self.policy(state)
            if quarterback_action is not None:
                actions["qb"] = quarterback_action
        state = self.play.step(actions)
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            self.finished += 1
            self.last_outcome = state.outcome.value
            if state.outcome is PlayOutcome.COMPLETED:
                self.completed += 1
            self._hold = _HOLD_TICKS
        return state
