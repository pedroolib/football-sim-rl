"""Single-receiver drive environment for Stage B (WR inside full drives).

The receiver is the learning agent; the quarterback uses the rule controller
unless a frozen policy is supplied, and the rest of the roster uses the M3 rule
controllers with the drive's sampled routes. The observation is the 40-float
receiver drive layout and the action is a 2-value movement vector.
"""

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
from qb_sim.rules.drive import Drive, DriveConfig
from qb_sim.rules.scenarios import STAGE_B_SCENARIO, Scenario
from qb_sim.training.randomization import AdaptationRanges, sample_adaptation_config
from qb_sim.training.rewards import (
    ReceiverDriveRewardConfig,
    progress_shaping,
    receiver_drive_reward,
    receiver_separation_potential,
)
from qb_sim.training.role_context_observations import (
    RECEIVER_DRIVE_OBSERVATION_SIZE,
    receiver_drive_observation,
)
from qb_sim.training.role_env import (
    MOVE_ACTION_SIZE,
    _decode_move,
    decide_player_action,
    inject_active_config,
)


class ReceiverDriveEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train one receiver inside drives against frozen teammates/opponents."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario | None = None,
        learner_id: str = "wr1",
        opponents: Mapping[str, Any] | None = None,
        reward_config: ReceiverDriveRewardConfig | None = None,
        drive_config: DriveConfig | None = None,
        goal_line: float = 100.0,
        max_steps: int = 2400,
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
        self.reward_config = (
            reward_config if reward_config is not None else ReceiverDriveRewardConfig()
        )
        self.drive_config = drive_config if drive_config is not None else DriveConfig()
        self.goal_line = float(goal_line)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(MOVE_ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(RECEIVER_DRIVE_OBSERVATION_SIZE,),
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
        self._separation_at_throw: float | None = None
        self._targeted_at_throw: bool | None = None
        self._separation_at_catch: float | None = None
        self._catch_x: float | None = None
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
        self._separation_at_throw = None
        self._targeted_at_throw = None
        self._separation_at_catch = None
        self._catch_x = None

    def rule_agent(self, player_id: str) -> BaseAgent:
        """Rule controller for a player, used to collect demonstrations."""
        return self._agents[player_id]

    def observation(self) -> np.ndarray:
        """Current learner observation with drive context."""
        index = self.play.receiver_ids.index(self.learner_id)
        shift = self.play.line_of_scrimmage - self.play.config.quarterback_start[0]
        start = vec2(
            self.play.config.receiver_starts[index][0] + shift,
            self.play.config.receiver_starts[index][1],
        )
        return receiver_drive_observation(
            self.play.state,
            self.learner_id,
            self._active_config,
            self.drive.routes[index],
            self.drive.state,
            self.goal_line,
            start=start,
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
        x_before = (
            float(self.play.state.player(self.learner_id).position[0])
            if self.play.state.carrier_id == self.learner_id
            else None
        )
        state = self.play.step(engine_actions)
        self._steps += 1
        self._track_diagnostics(state)
        reward = 0.0
        if x_before is not None and state.carrier_id == self.learner_id:
            reward += (
                max(0.0, float(state.player(self.learner_id).position[0]) - x_before)
                * self.reward_config.receiver.carry_yard
            )
        terminated = False
        truncated = False
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            down_before = self.drive.state.down
            distance_before = self.drive.state.distance
            los_before = self.drive.state.los
            result = self.drive.advance(self.play)
            caught = state.carrier_id == self.learner_id
            reward += receiver_drive_reward(
                result.outcome,
                result.yards,
                caught,
                self._targeted_open(),
                self.reward_config,
            )
            reward += progress_shaping(
                result.spot - result.yards,
                result.spot,
                self.goal_line,
                self.reward_config.progress_scale,
                self.reward_config.progress_gamma,
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
                    "pressure_at_throw": self._pressure_at_throw,
                    "arrival_time": self._arrival_time,
                    "separation_allowed_throw": self._separation_at_throw,
                    "separation_allowed_catch": self._separation_at_catch,
                    "targeted_at_throw": self._targeted_at_throw,
                    "separation_at_throw": self._separation_at_throw,
                    "separation_at_catch": self._separation_at_catch,
                    "caught": caught,
                    "yards_after_catch": self._yards_after_catch(result, caught),
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
        if self.reward_config.separation_shaping:
            reward += self.reward_config.shaping_gamma * next_potential - potential_before
        if not terminated and self._steps >= self.max_steps:
            truncated = True
        return self.observation(), float(reward), terminated, truncated, self._info()

    def render(self) -> None:
        return None

    def _targeted_open(self) -> bool:
        return (
            bool(self._targeted_at_throw)
            and self._separation_at_throw is not None
            and self._separation_at_throw >= self.reward_config.separation_threshold
        )

    def _yards_after_catch(self, result, caught: bool) -> float:
        if not caught or self._catch_x is None:
            return 0.0
        return float(result.spot - self._catch_x)

    def _track_diagnostics(self, state) -> None:
        if self._pressure_at_throw is None and state.throw_time is not None:
            self._pressure_at_throw = float(state.pressure)
        if self._arrival_time is None and self.play.rusher_ids:
            quarterback = state.player("qb")
            for rusher_id in self.play.rusher_ids:
                rusher = state.player(rusher_id)
                distance = float(np.linalg.norm(rusher.position - quarterback.position))
                if distance <= self._active_config.tackle_radius:
                    self._arrival_time = float(state.elapsed_seconds)
                    break
        if self._separation_at_throw is None and state.throw_time is not None:
            self._separation_at_throw = self._learner_separation(state)
            target = state.ball.target
            if target is not None:
                nearest = min(
                    self.play.receiver_ids,
                    key=lambda player_id: float(
                        np.linalg.norm(state.player(player_id).position - target)
                    ),
                )
                self._targeted_at_throw = nearest == self.learner_id
            elif state.carrier_id is not None:
                self._targeted_at_throw = state.carrier_id == self.learner_id
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

    def _potential(self, state) -> float:
        if state.ball.status is not BallStatus.HELD or state.carrier_id == self.learner_id:
            return 0.0
        return receiver_separation_potential(
            self._learner_separation(state),
            self.reward_config.shaping_scale,
        )

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
