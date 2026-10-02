"""Gymnasium environment for training the quarterback (M4/M5).

The QB is the learning agent; receivers and defenders use the rule-based
controllers from M3. Observations are normalized, role-local vectors and the
hybrid action is ``[move_x, move_y, aim_x, aim_y, power_raw, throw_trigger]``
with every component in ``[-1, 1]``. See ``training.actions`` for the mapping.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.loop import build_rule_agents
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import DEFAULT_SCENARIO, Scenario
from qb_sim.training.actions import ACTION_SIZE, decode_action
from qb_sim.training.observations import OBSERVATION_SIZE, observation_from_state
from qb_sim.training.rewards import RewardConfig, terminal_reward

__all__ = ["ACTION_SIZE", "OBSERVATION_SIZE", "QBPassingEnv"]


class QBPassingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train a quarterback against rule-based teammates and defenders."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 240,
    ) -> None:
        super().__init__()
        self.scenario = scenario if scenario is not None else DEFAULT_SCENARIO
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.play = Play(self.scenario.config)
        self._agents: dict[str, BaseAgent] = {}
        self._build_agents()
        self._steps = 0
        self._episode_seed = self.scenario.seed
        self.attempt_scale = 1.0

    def set_attempt_scale(self, scale: float) -> None:
        """Scale the attempt shaping, used to anneal it during training."""
        self.attempt_scale = max(0.0, float(scale))

    def _build_agents(self) -> None:
        self._agents = build_rule_agents(
            self.play,
            self.scenario.receiver_routes,
            self.scenario.config,
            include_quarterback=False,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new play; pass ``seed`` for a reproducible episode."""
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.play = Play(self.scenario.config, seed=self._episode_seed)
        self._build_agents()
        self._steps = 0
        return self._observation(), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply the QB action, advance one tick, and resolve the play."""
        engine_actions: dict[str, Move | Throw] = {}
        for player_id, agent in self._agents.items():
            agent_action = agent.decide(self.play.state)
            if agent_action is not None:
                engine_actions[player_id] = agent_action
        decoded = decode_action(action, self.play.state.ball.status)
        attempted = isinstance(decoded.get("qb"), Throw)
        elapsed_before = self.play.state.elapsed_seconds
        engine_actions.update(decoded)

        state = self.play.step(engine_actions)
        self._steps += 1
        terminated = state.outcome is not PlayOutcome.IN_PROGRESS
        truncated = not terminated and self._steps >= self.max_steps
        reward = self._step_reward(state, terminated, attempted, elapsed_before)
        return self._observation(), reward, terminated, truncated, self._info()

    def render(self) -> None:
        """No rendering in M4; visual demos belong to M5."""
        return None

    def _step_reward(
        self,
        state,
        terminated: bool,
        attempted: bool,
        elapsed_before: float,
    ) -> float:
        reward = self.reward_config.time_penalty
        if attempted and elapsed_before >= self.reward_config.attempt_min_seconds - 1e-9:
            reward += self.reward_config.attempt * self.attempt_scale
        if terminated:
            yards = 0.0
            if state.carrier_id is not None:
                yards = float(
                    state.player(state.carrier_id).position[0]
                    - self.scenario.config.quarterback_start[0]
                )
            reward += terminal_reward(state.outcome, yards, self.reward_config)
        return float(reward)

    def _observation(self) -> np.ndarray:
        return observation_from_state(self.play.state, self.play.config)

    def _info(self) -> dict[str, Any]:
        state = self.play.state
        return {
            "episode_seed": self._episode_seed,
            "scenario": self.scenario.name,
            "outcome": state.outcome.value,
            "elapsed_seconds": state.elapsed_seconds,
            "pressure": state.pressure,
            "throw_time": state.throw_time,
            "carrier_id": state.carrier_id,
            "steps": self._steps,
        }
