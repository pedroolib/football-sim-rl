"""Single-learner role environments for Stage B (WR play level, CB/rusher drives)."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.loop import build_rule_agents
from qb_sim.agents.rule_controllers import HeuristicQuarterback
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus, PlayOutcome
from qb_sim.engine.geometry import vec2
from qb_sim.rules.drive import Drive, DriveConfig, DriveState
from qb_sim.rules.play import Play
from qb_sim.rules.routes import sample_routes
from qb_sim.rules.scenarios import STAGE_B_SCENARIO, Scenario
from qb_sim.training.randomization import AdaptationRanges, sample_adaptation_config
from qb_sim.training.rewards import (
    DefenderRewardConfig,
    ReceiverRewardConfig,
    defender_drive_reward,
    defender_potential,
    receiver_result_reward,
)
from qb_sim.training.role_context_observations import (
    DEFENDER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
    defender_context_observation,
    receiver_context_observation,
)

MOVE_ACTION_SIZE = 2


def defender_target(play: Play, defender_id: str) -> str | None:
    """Coverage target for a defender, shared by the role environments."""
    state = play.state
    if state.carrier_id is not None:
        return state.carrier_id
    if defender_id in play.rusher_ids:
        return "qb"
    coverage = [
        player_id for player_id in play.defender_ids if player_id not in set(play.rusher_ids)
    ]
    index = coverage.index(defender_id)
    return play.receiver_ids[index % len(play.receiver_ids)]


def decide_player_action(
    play: Play,
    routes: tuple,
    player_id: str,
    policy: Any,
    drive_state: DriveState | None = None,
) -> Move | Throw | None:
    """Call a rule agent or a context-aware policy with the right signature."""
    if isinstance(policy, BaseAgent):
        return policy.decide(play.state)
    if player_id == "qb":
        if drive_state is None:
            return policy(play.state)
        return policy(play.state, routes, drive_state)
    if player_id in play.receiver_ids:
        index = play.receiver_ids.index(player_id)
        shift = play.line_of_scrimmage - play.config.quarterback_start[0]
        start = vec2(
            play.config.receiver_starts[index][0] + shift,
            play.config.receiver_starts[index][1],
        )
        if getattr(policy, "drive_aware", False) and drive_state is not None:
            return policy(play.state, routes[index], start, drive_state=drive_state)
        return policy(play.state, routes[index], start)
    return policy(play.state, defender_target(play, player_id), drive_state)


def inject_active_config(opponents: Mapping[str, Any], config: SimulationConfig) -> None:
    """Push the episode config into opponents that embed their own config."""
    for opponent in opponents.values():
        setter = getattr(opponent, "set_config", None)
        if setter is not None:
            setter(config)


def _decode_move(action: np.ndarray) -> Move | None:
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (MOVE_ACTION_SIZE,):
        raise ValueError(f"action must have {MOVE_ACTION_SIZE} values")
    direction = np.array(values, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return None
    if length > 1.0:
        direction = direction / length
    return Move(direction=direction)


class RolePlayEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train one receiver inside a single play with random routes."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario | None = None,
        learner_id: str = "wr1",
        opponents: Mapping[str, Any] | None = None,
        reward_config: ReceiverRewardConfig | None = None,
        goal_line: float = 100.0,
        max_steps: int = 240,
        los_range: tuple[float, float] = (25.0, 50.0),
        randomization: AdaptationRanges | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario if scenario is not None else STAGE_B_SCENARIO
        self.learner_id = learner_id
        self.opponents = dict(opponents or {})
        if "qb" not in self.opponents:
            self.opponents["qb"] = HeuristicQuarterback("qb", self.scenario.config)
        self.randomization = randomization
        self._curriculum_progress = 0.0
        self._active_config = self.scenario.config
        self.reward_config = reward_config if reward_config is not None else ReceiverRewardConfig()
        self.goal_line = float(goal_line)
        low, high = los_range
        if not 0.0 <= low <= high < self.goal_line:
            raise ValueError("los_range must be inside the field")
        starts = (
            (self.scenario.config.quarterback_start,)
            + tuple(self.scenario.config.receiver_starts)
            + tuple(self.scenario.config.defender_starts)
            + tuple(self.scenario.config.rusher_starts)
        )
        floor = max(
            0.0,
            self.scenario.config.quarterback_start[0] - min(start[0] for start in starts),
        )
        if low < floor:
            raise ValueError("los_range is below the formation floor")
        self.los_range = (float(low), float(high))
        self.max_steps = int(max_steps)
        self._drive_context = DriveState(
            los=float(self.scenario.config.quarterback_start[0]),
            down=1,
            distance=10.0,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(MOVE_ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(RECEIVER_CONTEXT_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.play = Play(self.scenario.config, goal_line=self.goal_line)
        self.routes: tuple = ()
        self._agents: dict[str, BaseAgent] = {}
        self._steps = 0
        self._episode_seed = self.scenario.seed
        self._separation_at_throw: float | None = None
        self._separation_at_catch: float | None = None
        self._catch_x: float | None = None
        self._setup_episode()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._setup_episode()
        return self.observation(), self._info()

    def set_curriculum_progress(self, progress: float) -> None:
        """Set the [0, 1] curriculum phase used by the next episode sample."""
        self._curriculum_progress = min(max(float(progress), 0.0), 1.0)

    @property
    def active_config(self) -> SimulationConfig:
        """Configuration sampled for the current episode (base when disabled)."""
        return self._active_config

    def _sample_config(self) -> SimulationConfig:
        if self.randomization is None:
            return self.scenario.config
        return sample_adaptation_config(
            self.scenario.config,
            np.random.default_rng(self._episode_seed + 1_000_003),
            self._curriculum_progress,
            self.randomization,
        )

    def _setup_episode(self) -> None:
        config = self._sample_config()
        self._active_config = config
        inject_active_config(self.opponents, config)
        rng = np.random.default_rng(self._episode_seed)
        low, high = self.los_range
        los = float(rng.uniform(low, high))
        self.routes = sample_routes(rng, len(config.receiver_starts))
        self.play = Play(
            config,
            seed=self._episode_seed,
            line_of_scrimmage=los,
            goal_line=self.goal_line,
        )
        self._drive_context.los = los
        self._agents = build_rule_agents(
            self.play,
            self.routes,
            config,
            include_quarterback=False,
        )
        self._steps = 0
        self._separation_at_throw = None
        self._separation_at_catch = None
        self._catch_x = None

    def rule_agent(self, player_id: str) -> BaseAgent:
        """Rule controller for a player, used to collect demonstrations."""
        return self._agents[player_id]

    def observation(self) -> np.ndarray:
        """Current learner observation."""
        index = self.play.receiver_ids.index(self.learner_id)
        shift = self.play.line_of_scrimmage - self.play.config.quarterback_start[0]
        start = vec2(
            self.play.config.receiver_starts[index][0] + shift,
            self.play.config.receiver_starts[index][1],
        )
        route = self.routes[index]
        return receiver_context_observation(
            self.play.state,
            self.learner_id,
            self.play.config,
            route,
            self.goal_line,
            start=start,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        engine_actions: dict[str, Move | Throw] = {}
        actors: dict[str, Any] = {**self._agents, **self.opponents}
        for player_id, policy in actors.items():
            if player_id == self.learner_id:
                continue
            if player_id in self.opponents:
                decided = decide_player_action(
                    self.play,
                    self.routes,
                    player_id,
                    self.opponents[player_id],
                    self._drive_context,
                )
            else:
                decided = policy.decide(self.play.state)
            if decided is not None:
                engine_actions[player_id] = decided
        learner_action = _decode_move(action)
        if learner_action is not None:
            engine_actions[self.learner_id] = learner_action

        x_before = (
            float(self.play.state.player(self.learner_id).position[0])
            if self.play.state.carrier_id == self.learner_id
            else None
        )
        state = self.play.step(engine_actions)
        self._steps += 1
        self._track_diagnostics(state)
        terminated = state.outcome is not PlayOutcome.IN_PROGRESS
        truncated = not terminated and self._steps >= self.max_steps
        reward = 0.0
        if x_before is not None and state.carrier_id == self.learner_id:
            reward += (
                max(0.0, float(state.player(self.learner_id).position[0]) - x_before)
                * self.reward_config.carry_yard
            )
        if terminated:
            caught = state.carrier_id == self.learner_id
            reward += receiver_result_reward(
                state.outcome, self._carrier_yards(state), caught, self.reward_config
            )
        return self.observation(), float(reward), terminated, truncated, self._info()

    def render(self) -> None:
        return None

    def _carrier_yards(self, state) -> float:
        if state.carrier_id is None:
            return 0.0
        return float(
            state.player(state.carrier_id).position[0] - self.play.line_of_scrimmage
        )

    def _track_diagnostics(self, state) -> None:
        if self._separation_at_throw is None and state.throw_time is not None:
            self._separation_at_throw = self._learner_separation(state)
        if self._separation_at_catch is None and state.carrier_id == self.learner_id:
            self._separation_at_catch = self._learner_separation(state)
            self._catch_x = float(state.player(self.learner_id).position[0])

    def _learner_separation(self, state) -> float:
        learner = state.player(self.learner_id)
        rusher_ids = set(self.play.rusher_ids)
        defenders = [
            state.player(player_id).position
            for player_id in self.play.defender_ids
            if player_id not in rusher_ids
        ]
        if not defenders:
            return 0.0
        return float(
            min(np.linalg.norm(learner.position - defender) for defender in defenders)
        )

    def _info(self) -> dict[str, Any]:
        state = self.play.state
        caught = state.carrier_id == self.learner_id
        yards = self._carrier_yards(state)
        yards_after_catch = 0.0
        if caught and self._catch_x is not None:
            yards_after_catch = yards - (self._catch_x - self.play.line_of_scrimmage)
        return {
            "episode_seed": self._episode_seed,
            "outcome": state.outcome.value,
            "yards": yards,
            "yards_after_catch": yards_after_catch,
            "caught": caught,
            "separation_at_throw": self._separation_at_throw,
            "separation_at_catch": self._separation_at_catch,
            "throw_time": state.throw_time,
            "elapsed_seconds": state.elapsed_seconds,
        }


class RoleDriveEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train one defender inside drives against frozen teammates/opponents."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario | None = None,
        learner_id: str = "cb1",
        opponents: Mapping[str, Any] | None = None,
        reward_config: DefenderRewardConfig | None = None,
        drive_config: DriveConfig | None = None,
        goal_line: float = 100.0,
        max_steps: int = 2400,
        randomization: AdaptationRanges | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario if scenario is not None else STAGE_B_SCENARIO
        self.learner_id = learner_id
        self.opponents = dict(opponents or {})
        self.randomization = randomization
        self._curriculum_progress = 0.0
        self._active_config = self.scenario.config
        self.reward_config = (
            reward_config if reward_config is not None else DefenderRewardConfig()
        )
        self.drive_config = drive_config if drive_config is not None else DriveConfig()
        self.goal_line = float(goal_line)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(MOVE_ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(DEFENDER_CONTEXT_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.drive = Drive(self.scenario.config, self.drive_config)
        self.play = self.drive.start_play()
        self.play_history: list[dict[str, Any]] = []
        self._agents: dict[str, BaseAgent] = {}
        self._steps = 0
        self._episode_seed = self.scenario.seed
        self._pressure_at_throw: float | None = None
        self._arrival_time: float | None = None
        self._separation_allowed_throw: float | None = None
        self._separation_allowed_catch: float | None = None
        self._build_agents()

    def set_curriculum_progress(self, progress: float) -> None:
        """Set the [0, 1] curriculum phase used by the next episode sample."""
        self._curriculum_progress = min(max(float(progress), 0.0), 1.0)

    @property
    def active_config(self) -> SimulationConfig:
        """Configuration sampled for the current episode (base when disabled)."""
        return self._active_config

    def _sample_config(self) -> SimulationConfig:
        if self.randomization is None:
            return self.scenario.config
        return sample_adaptation_config(
            self.scenario.config,
            np.random.default_rng(self._episode_seed + 1_000_003),
            self._curriculum_progress,
            self.randomization,
        )

    def _build_agents(self) -> None:
        self._agents = build_rule_agents(
            self.play,
            self.drive.routes,
            self._active_config,
            include_quarterback=False,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._active_config = self._sample_config()
        inject_active_config(self.opponents, self._active_config)
        self.drive = Drive(self._active_config, self.drive_config, seed=self._episode_seed)
        self.play_history = []
        self._steps = 0
        self._start_play()
        return self.observation(), self._info()

    def _start_play(self) -> None:
        self.play = self.drive.start_play()
        self._build_agents()
        self._pressure_at_throw = None
        self._arrival_time = None
        self._separation_allowed_throw = None
        self._separation_allowed_catch = None

    def rule_agent(self, player_id: str) -> BaseAgent:
        return self._agents[player_id]

    def observation(self) -> np.ndarray:
        return defender_context_observation(
            self.play.state,
            self.learner_id,
            defender_target(self.play, self.learner_id),
            self._active_config,
            self.drive.state,
            self.goal_line,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.drive.state.terminated:
            raise RuntimeError("the drive episode is over; call reset()")
        engine_actions: dict[str, Move | Throw] = {}
        actors: dict[str, Any] = {**self._agents, **self.opponents}
        for player_id, policy in actors.items():
            if player_id == self.learner_id:
                continue
            if player_id in self.opponents:
                decided = decide_player_action(
                    self.play,
                    self.drive.routes,
                    player_id,
                    self.opponents[player_id],
                    self.drive.state,
                )
            else:
                decided = policy.decide(self.play.state)
            if decided is not None:
                engine_actions[player_id] = decided
        learner_action = _decode_move(action)
        if learner_action is not None:
            engine_actions[self.learner_id] = learner_action

        potential_before = self._potential(self.play.state)
        state = self.play.step(engine_actions)
        self._steps += 1
        self._track_diagnostics(state)
        reward = 0.0
        terminated = False
        truncated = False
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            down_before = self.drive.state.down
            distance_before = self.drive.state.distance
            los_before = self.drive.state.los
            result = self.drive.advance(self.play)
            reward = defender_drive_reward(
                result,
                self.reward_config.terminal,
                goal_line=self.goal_line,
                interception_bonus=self.reward_config.interception_bonus,
            )
            self.play_history.append(
                {
                    "down": down_before,
                    "distance": distance_before,
                    "los_before": los_before,
                    "outcome": result.outcome.value,
                    "spot": result.spot,
                    "yards": result.yards,
                    "first_down": result.first_down,
                    "touchdown": result.touchdown,
                    "throw_distance": result.throw_distance,
                    "sack": result.outcome is PlayOutcome.SACK,
                    "coverage_sack": result.coverage_sack,
                    "routes": [route.value for route in self.drive.routes],
                    "pressure_at_throw": self._pressure_at_throw,
                    "arrival_time": self._arrival_time,
                    "separation_allowed_throw": self._separation_allowed_throw,
                    "separation_allowed_catch": self._separation_allowed_catch,
                }
            )
            terminated = self.drive.state.terminated
            if not terminated:
                self._start_play()
                next_potential = self._potential(self.play.state)
            else:
                next_potential = 0.0
        else:
            next_potential = self._potential(state)
        if self.reward_config.shaping:
            reward += (
                self.reward_config.shaping_gamma * next_potential - potential_before
            )
        if not terminated and self._steps >= self.max_steps:
            truncated = True
        return self.observation(), float(reward), terminated, truncated, self._info()

    def render(self) -> None:
        return None

    def _track_diagnostics(self, state) -> None:
        if self._pressure_at_throw is None and state.throw_time is not None:
            self._pressure_at_throw = float(state.pressure)
        if self.learner_id in self.play.rusher_ids:
            if self._arrival_time is None:
                quarterback = state.player("qb")
                rusher = state.player(self.learner_id)
                distance_to_quarterback = float(
                    np.linalg.norm(rusher.position - quarterback.position)
                )
                if distance_to_quarterback <= self._active_config.tackle_radius:
                    self._arrival_time = float(state.elapsed_seconds)
            return
        if self.learner_id not in self.play.defender_ids:
            return
        target_id = defender_target(self.play, self.learner_id)
        if target_id is None:
            return
        if self._separation_allowed_throw is None and state.throw_time is not None:
            self._separation_allowed_throw = self._target_separation(state, target_id)
        if self._separation_allowed_catch is None and state.carrier_id is not None:
            self._separation_allowed_catch = self._target_separation(state, target_id)

    def _target_separation(self, state, target_id: str) -> float:
        learner = state.player(self.learner_id)
        target = state.player(target_id)
        return float(np.linalg.norm(learner.position - target.position))

    def _potential(self, state) -> float:
        ball = state.ball
        if ball.status is BallStatus.IN_FLIGHT:
            distance = float(
                np.linalg.norm(state.player(self.learner_id).position - ball.position)
            )
            return defender_potential(distance, self.reward_config.shaping_scale)
        target_id = defender_target(self.play, self.learner_id)
        if target_id is None:
            return 0.0
        distance = float(
            np.linalg.norm(
                state.player(self.learner_id).position - state.player(target_id).position
            )
        )
        return defender_potential(distance, self.reward_config.shaping_scale)

    def _info(self) -> dict[str, Any]:
        return {
            "episode_seed": self._episode_seed,
            "down": self.drive.state.down,
            "distance": self.drive.state.distance,
            "los": self.drive.state.los,
            "outcome": self.play.state.outcome.value,
            "drive_end_reason": self.drive.state.end_reason,
            "steps": self._steps,
            "plays": list(self.play_history),
        }
