"""Monitoring helpers for live training dashboards (no pygame, no torch)."""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from qb_sim.agents.loop import build_rule_agents
from qb_sim.agents.rule_controllers import HeuristicQuarterback, ReceiverRouteController
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayOutcome, PlayState
from qb_sim.engine.geometry import Vector2, vec2
from qb_sim.rules.drive import Drive, DriveConfig, DriveState
from qb_sim.rules.play import Play
from qb_sim.rules.routes import Route, RouteName, sample_routes
from qb_sim.rules.scenarios import Scenario
from qb_sim.training.role_env import decide_player_action, defender_target

QuarterbackPolicy = Callable[[PlayState], Move | Throw | None]
DriveQuarterbackPolicy = Callable[
    [PlayState, tuple[RouteName, ...], DriveState],
    Move | Throw | None,
]

_CHECKPOINT_SUFFIX = "_steps.zip"
_HOLD_TICKS = 10
_ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}



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


def checkpoint_catalog(run_dir: str | Path) -> list[Path]:
    """Return the selectable policy generations, ordered by training step."""
    root = Path(run_dir)
    directory = root / "checkpoints"
    catalog: list[Path] = []
    if directory.is_dir():
        catalog.extend(sorted(directory.glob(f"*{_CHECKPOINT_SUFFIX}"), key=checkpoint_steps))
    for relative in ("best/best_model.zip", "final_model.zip"):
        path = root / relative
        if path.is_file():
            catalog.append(path)
    return catalog


def checkpoint_label(path: str | Path) -> str:
    """Human-readable label for one catalog entry."""
    stem = Path(path).stem
    if stem == "best_model":
        return "best"
    if stem == "final_model":
        return "final"
    steps = checkpoint_steps(path)
    return f"step {steps:,}" if steps >= 0 else stem


def total_timesteps(run_dir: str | Path) -> int | None:
    """Read the planned timestep budget from the run ``config.json``."""
    path = Path(run_dir) / "config.json"
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = metadata.get("total_timesteps")
    return int(value) if value is not None else None



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


def rule_drive_quarterback_policy(config: SimulationConfig) -> DriveQuarterbackPolicy:
    """A drive demo policy using the M3 heuristic quarterback."""
    agent = HeuristicQuarterback("qb", config)

    def policy(
        state: PlayState,
        routes: tuple[RouteName, ...],
        drive_state: DriveState,
    ) -> Move | Throw | None:
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


@dataclass(slots=True)
class DriveDemoPanel:
    """One live demo drive with rule teammates and a swappable QB policy."""

    scenario: Scenario
    policy: DriveQuarterbackPolicy | None = None
    seed: int = 0
    drive_config: DriveConfig | None = None
    drives: int = 0
    touchdowns: int = 0
    points: float = 0.0
    first_downs: int = 0
    last_outcome: str = ""
    _hold: int = field(default=0, repr=False)
    _drive: Drive = field(init=False, repr=False)
    _play: Play = field(init=False, repr=False)
    _agents: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._drive = Drive(self.scenario.config, self.drive_config, seed=self.seed)
        self._start_play()

    @property
    def play(self) -> Play:
        """The current mechanical play."""
        return self._play

    @property
    def drive(self) -> Drive:
        """The current drive state holder."""
        return self._drive

    @property
    def receiver_routes(self) -> dict[str, Route]:
        """Routes of the rule receivers, for visualization."""
        return {
            player_id: agent.route
            for player_id, agent in self._agents.items()
            if isinstance(agent, ReceiverRouteController)
        }

    @property
    def touchdown_pct(self) -> float:
        """Touchdowns as a percentage of finished drives."""
        return 100.0 * self.touchdowns / self.drives if self.drives else 0.0

    @property
    def points_per_drive(self) -> float:
        """Average points per finished drive."""
        return self.points / self.drives if self.drives else 0.0

    @property
    def first_downs_per_drive(self) -> float:
        """Average first downs per finished drive."""
        return self.first_downs / self.drives if self.drives else 0.0

    @property
    def down_label(self) -> str:
        """Down, distance, and field position of the current play."""
        state = self._drive.state
        if state.terminated:
            return state.end_reason or "drive over"
        ordinal = _ORDINALS.get(state.down, f"{state.down}th")
        return f"{ordinal} & {state.distance:.0f} @ {state.los:.0f}"

    def set_policy(self, policy: DriveQuarterbackPolicy | None) -> None:
        """Swap the quarterback policy without resetting the drive."""
        self.policy = policy

    def restart(self, seed: int | None = None) -> PlayState:
        """Start a new drive with the next seed."""
        if seed is not None:
            self.seed = seed
        self._drive = Drive(self.scenario.config, self.drive_config, seed=self.seed)
        self.seed += 1
        self._hold = 0
        self._start_play()
        return self._play.state

    def _start_play(self) -> None:
        self._play = self._drive.start_play()
        self._agents = build_rule_agents(
            self._play,
            self._drive.routes,
            self.scenario.config,
            include_quarterback=False,
        )

    def tick(self) -> PlayState:
        """Advance one tick; start the next down or hold after the drive ends."""
        state = self._play.state
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            if self._hold > 0:
                self._hold -= 1
                return state
            self.restart()
            return self._play.state

        actions: dict[str, Move | Throw] = {}
        for player_id, agent in self._agents.items():
            action = agent.decide(state)
            if action is not None:
                actions[player_id] = action
        if self.policy is not None:
            quarterback_action = self.policy(state, self._drive.routes, self._drive.state)
            if quarterback_action is not None:
                actions["qb"] = quarterback_action
        state = self._play.step(actions)
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            result = self._drive.advance(self._play)
            self.first_downs += int(result.first_down)
            if self._drive.state.terminated:
                self.drives += 1
                self.touchdowns += int(result.touchdown)
                self.points += result.points
                self.last_outcome = self._drive.state.end_reason or result.outcome.value
                self._hold = _HOLD_TICKS
            else:
                self.last_outcome = result.outcome.value
                self._start_play()
        return state


@dataclass(slots=True)
class RolePlayDemoPanel:
    """One live demo play for a receiver policy with rule or frozen teammates."""

    scenario: Scenario
    learner_id: str = "wr1"
    policy: Callable | None = None
    opponents: dict = field(default_factory=dict)
    seed: int = 0
    goal_line: float = 100.0
    plays: int = 0
    catches: int = 0
    yards: float = 0.0
    last_outcome: str = ""
    _hold: int = field(default=0, repr=False)
    _play: Play = field(init=False, repr=False)
    _routes: tuple[RouteName, ...] = field(init=False, repr=False)
    _agents: dict = field(default_factory=dict, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _drive_context: DriveState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._drive_context = DriveState(
            los=float(self.scenario.config.quarterback_start[0]),
            down=1,
            distance=10.0,
        )
        self._setup_episode()

    @property
    def play(self) -> Play:
        """The current mechanical play."""
        return self._play

    @property
    def routes(self) -> tuple[RouteName, ...]:
        """Routes sampled for the current play."""
        return self._routes

    @property
    def receiver_routes(self) -> dict[str, Route]:
        """Routes of the rule receivers, for visualization."""
        return {
            player_id: agent.route
            for player_id, agent in self._agents.items()
            if isinstance(agent, ReceiverRouteController)
        }

    @property
    def catch_pct(self) -> float:
        """Learner catches as a percentage of finished plays."""
        return 100.0 * self.catches / self.plays if self.plays else 0.0

    def set_policy(self, policy: Callable | None) -> None:
        """Swap the receiver policy without resetting the play."""
        self.policy = policy

    def _setup_episode(self) -> None:
        self._routes = sample_routes(self._rng, len(self.scenario.config.receiver_starts))
        play_seed = int(self._rng.integers(0, 2**31 - 1))
        self._play = Play(self.scenario.config, seed=play_seed, goal_line=self.goal_line)
        self._agents = build_rule_agents(
            self._play,
            self._routes,
            self.scenario.config,
            include_quarterback=False,
        )
        self._hold = 0

    def _learner_start(self) -> Vector2:
        index = self._play.receiver_ids.index(self.learner_id)
        start = self.scenario.config.receiver_starts[index]
        shift = self._play.line_of_scrimmage - self.scenario.config.quarterback_start[0]
        return vec2(start[0] + shift, start[1])

    def restart(self, seed: int | None = None) -> PlayState:
        """Start a new play with a fresh random route pair."""
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._setup_episode()
        return self._play.state

    def tick(self) -> PlayState:
        """Advance one simulation tick; auto-restart after a short hold."""
        state = self._play.state
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            if self._hold > 0:
                self._hold -= 1
                return state
            self.restart(seed=self.seed + 1)
            return self._play.state

        actions: dict[str, Move | Throw] = {}
        for player_id, agent in self._agents.items():
            if player_id == self.learner_id:
                continue
            if player_id in self.opponents:
                decided = decide_player_action(
                    self._play,
                    self._routes,
                    player_id,
                    self.opponents[player_id],
                    self._drive_context,
                )
            else:
                decided = agent.decide(state)
            if decided is not None:
                actions[player_id] = decided
        if self.policy is not None:
            index = self._play.receiver_ids.index(self.learner_id)
            decided = self.policy(state, self._routes[index], self._learner_start())
        else:
            decided = (
                self._agents[self.learner_id].decide(state)
                if self.learner_id in self._agents
                else None
            )
        if decided is not None:
            actions[self.learner_id] = decided
        state = self._play.step(actions)
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            self.plays += 1
            self.last_outcome = state.outcome.value
            if state.carrier_id == self.learner_id:
                self.catches += 1
                self.yards += float(
                    state.player(self.learner_id).position[0]
                    - self.scenario.config.quarterback_start[0]
                )
            self._hold = _HOLD_TICKS
        return state


@dataclass(slots=True)
class RoleDriveDemoPanel:
    """One live demo drive for a defender policy with rule or frozen teammates."""

    scenario: Scenario
    learner_id: str = "cb1"
    policy: Callable | None = None
    opponents: dict = field(default_factory=dict)
    seed: int = 0
    drive_config: DriveConfig | None = None
    drives: int = 0
    touchdowns_allowed: int = 0
    sacks: int = 0
    last_outcome: str = ""
    _hold: int = field(default=0, repr=False)
    _drive: Drive = field(init=False, repr=False)
    _play: Play = field(init=False, repr=False)
    _agents: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._drive = Drive(self.scenario.config, self.drive_config, seed=self.seed)
        self._start_play()

    @property
    def play(self) -> Play:
        """The current mechanical play."""
        return self._play

    @property
    def drive(self) -> Drive:
        """The current drive state holder."""
        return self._drive

    @property
    def receiver_routes(self) -> dict[str, Route]:
        """Routes of the rule receivers, for visualization."""
        return {
            player_id: agent.route
            for player_id, agent in self._agents.items()
            if isinstance(agent, ReceiverRouteController)
        }

    @property
    def touchdown_pct(self) -> float:
        """Touchdowns allowed as a percentage of finished drives."""
        return (
            100.0 * self.touchdowns_allowed / self.drives if self.drives else 0.0
        )

    def set_policy(self, policy: Callable | None) -> None:
        """Swap the defender policy without resetting the drive."""
        self.policy = policy

    @property
    def down_label(self) -> str:
        """Down, distance, and field position of the current play."""
        state = self._drive.state
        if state.terminated:
            return state.end_reason or "drive over"
        ordinal = _ORDINALS.get(state.down, f"{state.down}th")
        return f"{ordinal} & {state.distance:.0f} @ {state.los:.0f}"

    def _start_play(self) -> None:
        self._play = self._drive.start_play()
        self._agents = build_rule_agents(
            self._play,
            self._drive.routes,
            self.scenario.config,
            include_quarterback=False,
        )
        self._hold = 0

    def restart(self, seed: int | None = None) -> PlayState:
        """Start a new drive with the next seed."""
        if seed is not None:
            self.seed = int(seed)
        self._drive = Drive(self.scenario.config, self.drive_config, seed=self.seed)
        self._start_play()
        return self._play.state

    def tick(self) -> PlayState:
        """Advance one tick; start the next down or hold after the drive ends."""
        state = self._play.state
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            if self._hold > 0:
                self._hold -= 1
                return state
            self.restart(seed=self.seed + 1)
            return self._play.state

        actions: dict[str, Move | Throw] = {}
        for player_id, agent in self._agents.items():
            if player_id == self.learner_id:
                continue
            if player_id in self.opponents:
                decided = decide_player_action(
                    self._play,
                    self._drive.routes,
                    player_id,
                    self.opponents[player_id],
                    self._drive.state,
                )
            else:
                decided = agent.decide(state)
            if decided is not None:
                actions[player_id] = decided
        if self.policy is not None:
            decided = self.policy(
                state,
                defender_target(self._play, self.learner_id),
                self._drive.state,
            )
        else:
            decided = (
                self._agents[self.learner_id].decide(state)
                if self.learner_id in self._agents
                else None
            )
        if decided is not None:
            actions[self.learner_id] = decided
        state = self._play.step(actions)
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            result = self._drive.advance(self._play)
            self.last_outcome = self._drive.state.end_reason or result.outcome.value
            if self._drive.state.terminated:
                self.drives += 1
                self.touchdowns_allowed += int(result.touchdown)
                self._hold = _HOLD_TICKS
            else:
                self._start_play()
        return state
