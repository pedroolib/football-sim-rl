"""Gymnasium environment whose episode is a 3-down drive (M7).

The quarterback is the learning agent; receivers, corners, and rushers use the
M3 rule controllers with the drive's sampled routes. The observation is the
61-float drive layout and the action is the same 6-value hybrid action.
"""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.loop import build_rule_agents
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.drive import Drive, DriveConfig
from qb_sim.rules.scenarios import RUSHED_SCENARIO, Scenario
from qb_sim.training.actions import ACTION_SIZE, decode_action
from qb_sim.training.drive_observations import (
    DRIVE_OBSERVATION_SIZE,
    drive_observation_from_state,
)
from qb_sim.training.randomization import AdaptationRanges, sample_adaptation_config
from qb_sim.training.rewards import DriveRewardConfig, drive_result_reward
from qb_sim.training.role_env import decide_player_action, inject_active_config


class QBDriveEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train a quarterback to move the chains and score touchdowns."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario | None = None,
        drive_config: DriveConfig | None = None,
        reward_config: DriveRewardConfig | None = None,
        max_episode_steps: int = 2400,
        opponents: Mapping[str, Any] | None = None,
        randomization: AdaptationRanges | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario if scenario is not None else RUSHED_SCENARIO
        self.randomization = randomization
        self._curriculum_progress = 0.0
        self._active_config = self.scenario.config
        self.drive_config = drive_config if drive_config is not None else DriveConfig()
        self.reward_config = reward_config if reward_config is not None else DriveRewardConfig()
        self.max_episode_steps = int(max_episode_steps)
        self.opponents = dict(opponents or {})
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(DRIVE_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.drive = Drive(self.scenario.config, self.drive_config)
        self.play = self.drive.start_play()
        self.play_history: list[dict[str, Any]] = []
        self._agents: dict[str, BaseAgent] = {}
        self._build_agents()
        self._steps = 0
        self._episode_seed = self.scenario.seed
        self._drive_end_reason: str | None = None
        self.attempt_scale = 1.0

    def set_attempt_scale(self, scale: float) -> None:
        """Scale the attempt shaping, used to anneal it during training."""
        self.attempt_scale = max(0.0, float(scale))

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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new drive; pass ``seed`` for a reproducible episode."""
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
        self._drive_end_reason = None
        self._start_play()
        return self._observation(), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply the QB action and advance the play, starting the next down if needed."""
        if self.drive.state.terminated:
            raise RuntimeError("the drive episode is over; call reset()")
        engine_actions: dict[str, Move | Throw] = {}
        for player_id, agent in self._agents.items():
            if player_id in self.opponents:
                agent_action = decide_player_action(
                    self.play,
                    self.drive.routes,
                    player_id,
                    self.opponents[player_id],
                    self.drive.state,
                )
            else:
                agent_action = agent.decide(self.play.state)
            if agent_action is not None:
                engine_actions[player_id] = agent_action
        decoded = decode_action(action, self.play.state.ball.status)
        attempted = isinstance(decoded.get("qb"), Throw)
        elapsed_before = self.play.state.elapsed_seconds
        engine_actions.update(decoded)

        terminated = False
        truncated = False
        reward = self.reward_config.play.time_penalty
        if attempted and elapsed_before >= self.reward_config.play.attempt_min_seconds - 1e-9:
            reward += self.reward_config.play.attempt * self.attempt_scale

        state = self.play.step(engine_actions)
        self._steps += 1
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            down_before = self.drive.state.down
            distance_before = self.drive.state.distance
            los_before = self.drive.state.los
            result = self.drive.advance(self.play)
            self._drive_end_reason = result.end_reason
            reward += drive_result_reward(result, self.reward_config)
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
                }
            )
            terminated = self.drive.state.terminated
            if not terminated:
                self._start_play()
        if not terminated and self._steps >= self.max_episode_steps:
            truncated = True
        return self._observation(), float(reward), terminated, truncated, self._info()

    def render(self) -> None:
        """No rendering for the drive training environment."""
        return None

    def _start_play(self) -> None:
        self.play = self.drive.start_play()
        self._build_agents()

    def _build_agents(self) -> None:
        self._agents = build_rule_agents(
            self.play,
            self.drive.routes,
            self._active_config,
            include_quarterback=False,
        )
        self._agents.update(self.opponents)

    def _observation(self) -> np.ndarray:
        return drive_observation_from_state(
            state=self.play.state,
            routes=self.drive.routes,
            drive_state=self.drive.state,
            config=self._active_config,
            goal_line=self.drive_config.goal_line,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "episode_seed": self._episode_seed,
            "scenario": self.scenario.name,
            "down": self.drive.state.down,
            "distance": self.drive.state.distance,
            "los": self.drive.state.los,
            "play_number": self.drive.state.play_number,
            "points": self.drive.state.points,
            "first_downs": self.drive.state.first_downs,
            "outcome": self.play.state.outcome.value,
            "drive_end_reason": self._drive_end_reason,
            "throw_time": self.play.state.throw_time,
            "steps": self._steps,
            "plays": list(self.play_history),
        }
