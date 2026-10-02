"""Cycle through scenarios so one policy trains across rosters."""

from typing import Any

import gymnasium as gym
import numpy as np

from qb_sim.rules.scenarios import Scenario, get_scenario
from qb_sim.training.env import QBPassingEnv


class MultiScenarioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Wrap :class:`QBPassingEnv`, cycling the scenario on every reset."""

    metadata = {"render_modes": []}

    def __init__(self, scenario_names: tuple[str, ...], max_steps: int = 240) -> None:
        super().__init__()
        if not scenario_names:
            raise ValueError("at least one scenario is required")
        self.scenario_names = tuple(scenario_names)
        self.max_steps = int(max_steps)
        self._index = 0
        self._attempt_scale = 1.0
        self._env = QBPassingEnv(get_scenario(self.scenario_names[0]), max_steps=self.max_steps)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    @property
    def scenario(self) -> Scenario:
        """Scenario of the current episode."""
        return self._env.scenario

    @property
    def attempt_scale(self) -> float:
        """Attempt-shaping scale applied to the current episode."""
        return self._attempt_scale

    def set_attempt_scale(self, scale: float) -> None:
        """Persist and forward the annealing scale to the current episode."""
        self._attempt_scale = max(0.0, float(scale))
        self._env.set_attempt_scale(self._attempt_scale)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        name = self.scenario_names[self._index % len(self.scenario_names)]
        self._index += 1
        self._env = QBPassingEnv(get_scenario(name), max_steps=self.max_steps)
        self._env.set_attempt_scale(self._attempt_scale)
        return self._env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._env.step(action)

    def render(self) -> None:
        """No rendering for training environments."""
        return None

    def close(self) -> None:
        self._env.close()
