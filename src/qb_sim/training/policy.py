"""Adapt trained Stable-Baselines3 policies to the engine's player hooks.

Stable-Baselines3 and PyTorch are imported lazily inside the loader
functions, so importing this module never loads them.
"""

from pathlib import Path
from typing import Any

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayState
from qb_sim.engine.geometry import Vector2
from qb_sim.rules.drive import DriveState
from qb_sim.rules.routes import RouteName
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.actions import decode_action
from qb_sim.training.drive_observations import drive_observation_from_state
from qb_sim.training.observations import observation_from_state
from qb_sim.training.role_context_observations import (
    defender_context_observation,
    receiver_context_observation,
)


class PolicyQuarterback:
    """Callable ``(PlayState) -> Move | Throw | None`` backed by a PPO model."""

    def __init__(self, model: Any, config: SimulationConfig) -> None:
        self._model = model
        self._config = config

    def __call__(self, state: PlayState) -> Move | Throw | None:
        observation = observation_from_state(state, self._config)
        action, _ = self._model.predict(observation, deterministic=True)
        return decode_action(np.asarray(action, dtype=np.float32), state.ball.status).get("qb")


class DrivePolicyQuarterback:
    """Drive-aware callable backed by a PPO model trained on the 57-float layout."""

    def __init__(
        self,
        model: Any,
        config: SimulationConfig,
        goal_line: float = 100.0,
    ) -> None:
        self._model = model
        self._config = config
        self._goal_line = float(goal_line)

    def __call__(
        self,
        state: PlayState,
        routes: tuple[RouteName, ...],
        drive_state: DriveState,
    ) -> Move | Throw | None:
        observation = drive_observation_from_state(
            state,
            routes,
            drive_state,
            self._config,
            self._goal_line,
        )
        action, _ = self._model.predict(observation, deterministic=True)
        return decode_action(np.asarray(action, dtype=np.float32), state.ball.status).get("qb")


def _move_from_action(action: Any) -> Move | None:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    direction = np.array(values[:2], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return None
    if length > 1.0:
        direction = direction / length
    return Move(direction=direction)


class ReceiverPolicy:
    """Receiver callable ``(state, route, start=None) -> Move | None``."""

    def __init__(
        self,
        model: Any,
        config: SimulationConfig,
        player_id: str,
        goal_line: float = 100.0,
    ) -> None:
        self._model = model
        self._config = config
        self._player_id = player_id
        self._goal_line = float(goal_line)

    def __call__(
        self,
        state: PlayState,
        route: RouteName | None,
        start: Vector2 | None = None,
    ) -> Move | None:
        observation = receiver_context_observation(
            state,
            self._player_id,
            self._config,
            route,
            self._goal_line,
            start,
        )
        action, _ = self._model.predict(observation, deterministic=True)
        return _move_from_action(action)


class DefenderPolicy:
    """Defender callable ``(state, target_id, drive_state) -> Move | None``."""

    def __init__(
        self,
        model: Any,
        config: SimulationConfig,
        player_id: str,
        goal_line: float = 100.0,
    ) -> None:
        self._model = model
        self._config = config
        self._player_id = player_id
        self._goal_line = float(goal_line)

    def __call__(
        self,
        state: PlayState,
        target_id: str | None,
        drive_state: DriveState,
    ) -> Move | None:
        observation = defender_context_observation(
            state,
            self._player_id,
            target_id,
            self._config,
            drive_state,
            self._goal_line,
        )
        action, _ = self._model.predict(observation, deterministic=True)
        return _move_from_action(action)


def load_policy(path: str | Path, scenario_name: str = "default") -> PolicyQuarterback:
    """Load a saved PPO model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="auto")
    return PolicyQuarterback(model, scenario.config)


def load_drive_policy(path: str | Path, scenario_name: str = "rushed") -> DrivePolicyQuarterback:
    """Load a saved PPO drive model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="auto")
    return DrivePolicyQuarterback(model, scenario.config)


def load_receiver_policy(
    path: str | Path,
    scenario_name: str = "stage_b",
    player_id: str = "wr1",
) -> ReceiverPolicy:
    """Load a saved PPO receiver model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="auto")
    return ReceiverPolicy(model, scenario.config, player_id)


def load_defender_policy(
    path: str | Path,
    scenario_name: str = "stage_b",
    player_id: str = "cb1",
) -> DefenderPolicy:
    """Load a saved PPO defender model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="auto")
    return DefenderPolicy(model, scenario.config, player_id)
