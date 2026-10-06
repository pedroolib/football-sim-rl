"""Adapt trained Stable-Baselines3 policies to the engine's player hooks.

Stable-Baselines3 and PyTorch are imported lazily inside the loader
functions, so importing this module never loads them.
"""

from pathlib import Path
from typing import Any

import numpy as np

from qb_sim.agents.base import BaseAgent
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayState
from qb_sim.engine.geometry import Vector2
from qb_sim.rules.drive import DriveState
from qb_sim.rules.routes import RouteName
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.actions import decode_action
from qb_sim.training.drive_observations import drive_observation_from_state
from qb_sim.training.observations import OBSERVATION_SIZE, observation_from_state
from qb_sim.training.role_context_observations import (
    BALL_LANDING_SLOTS,
    RECEIVER_DRIVE_OBSERVATION_SIZE,
    defender_context_observation,
    receiver_context_observation,
    receiver_drive_observation,
)

SUPPORTED_PLAY_SIZES = (OBSERVATION_SIZE,)
SUPPORTED_DRIVE_SIZES = (62, 61, 57)
SUPPORTED_RECEIVER_SIZES = (43, 40, 38, 35, 31)
SUPPORTED_DEFENDER_SIZES = (31, 27, 24)


def _model_observation_size(model: Any) -> int | None:
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    return int(shape[0]) if shape else None


def _adapt_observation(observation: np.ndarray, expected: int | None) -> np.ndarray:
    """Slice current observations down to an older model's layout.

    Capability features are appended at the end, so historical layouts are a
    prefix of the current observation.
    """
    if expected is None or observation.shape[0] == expected:
        return observation
    if observation.shape[0] > expected:
        return observation[:expected]
    raise ValueError(
        f"el modelo espera {expected} valores de observación y la versión actual "
        f"genera {observation.shape[0]}"
    )


def _ensure_observation_size(
    model: Any,
    supported: tuple[int, ...],
    role: str,
    path: str | Path,
) -> None:
    """Reject models trained against an unknown observation layout."""
    actual = _model_observation_size(model)
    if actual not in supported:
        sizes = ", ".join(str(size) for size in supported)
        raise ValueError(
            f"{role}: el modelo {path} espera {actual} valores de observación; "
            f"tamaños soportados: {sizes}."
        )


class MixedPolicy:
    """Sample one frozen opponent policy per play from a mixture.

    The engine creates a fresh ``PlayState`` for every play and mutates that
    same object on each ``step``, so tracking the state object's identity lets
    the mixture stick with one sub-policy for the whole play and resample on
    the next one.
    """

    def __init__(self, policies: tuple[Any, ...], seed: int = 0) -> None:
        if not policies:
            raise ValueError("MixedPolicy necesita al menos una política")
        self.policies = tuple(policies)
        self._rng = np.random.default_rng(seed)
        self._last_state: Any = None
        self._active: Any = None

    def _active_policy(self, state: Any) -> Any:
        if state is not self._last_state:
            self._last_state = state
            self._active = self.policies[int(self._rng.integers(0, len(self.policies)))]
        return self._active

    def __call__(self, state: Any, *args: Any, **kwargs: Any) -> Any:
        policy = self._active_policy(state)
        if isinstance(policy, BaseAgent):
            return policy.decide(state)
        return policy(state, *args, **kwargs)

    def set_config(self, config: SimulationConfig) -> None:
        """Forward the episode config to every sub-policy that supports it."""
        for policy in self.policies:
            setter = getattr(policy, "set_config", None)
            if setter is not None:
                setter(config)

    def reset(self) -> None:
        """Forget the cached play so the next call samples a fresh sub-policy."""
        self._last_state = None
        self._active = None


class PolicyQuarterback:
    """Callable ``(PlayState) -> Move | Throw | None`` backed by a PPO model."""

    def __init__(self, model: Any, config: SimulationConfig) -> None:
        self._model = model
        self._config = config
        self._expected_size = _model_observation_size(model)

    def set_config(self, config: SimulationConfig) -> None:
        """Replace the embedded config (used when the episode is randomized)."""
        self._config = config

    def __call__(self, state: PlayState) -> Move | Throw | None:
        observation = _adapt_observation(
            observation_from_state(state, self._config),
            self._expected_size,
        )
        action, _ = self._model.predict(observation, deterministic=True)
        return decode_action(np.asarray(action, dtype=np.float32), state.ball.status).get("qb")


class DrivePolicyQuarterback:
    """Drive-aware callable backed by a PPO model trained on the 62-float layout."""

    def __init__(
        self,
        model: Any,
        config: SimulationConfig,
        goal_line: float = 100.0,
    ) -> None:
        self._model = model
        self._config = config
        self._goal_line = float(goal_line)
        self._expected_size = _model_observation_size(model)

    def set_config(self, config: SimulationConfig) -> None:
        """Replace the embedded config (used when the episode is randomized)."""
        self._config = config

    def __call__(
        self,
        state: PlayState,
        routes: tuple[RouteName, ...],
        drive_state: DriveState,
    ) -> Move | Throw | None:
        observation = _adapt_observation(
            drive_observation_from_state(
                state,
                routes,
                drive_state,
                self._config,
                self._goal_line,
            ),
            self._expected_size,
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
        self._expected_size = _model_observation_size(model)

    def set_config(self, config: SimulationConfig) -> None:
        """Replace the embedded config (used when the episode is randomized)."""
        self._config = config

    @property
    def drive_aware(self) -> bool:
        """Whether the model was trained on a drive layout (43 or 40 values)."""
        return self._expected_size in (
            RECEIVER_DRIVE_OBSERVATION_SIZE,
            RECEIVER_DRIVE_OBSERVATION_SIZE - BALL_LANDING_SLOTS,
        )

    def __call__(
        self,
        state: PlayState,
        route: RouteName | None,
        start: Vector2 | None = None,
        drive_state: DriveState | None = None,
    ) -> Move | None:
        if drive_state is not None and self.drive_aware:
            observation = receiver_drive_observation(
                state,
                self._player_id,
                self._config,
                route,
                drive_state,
                self._goal_line,
                start,
            )
        else:
            observation = receiver_context_observation(
                state,
                self._player_id,
                self._config,
                route,
                self._goal_line,
                start,
            )
        observation = _adapt_observation(observation, self._expected_size)
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
        self._expected_size = _model_observation_size(model)

    def set_config(self, config: SimulationConfig) -> None:
        """Replace the embedded config (used when the episode is randomized)."""
        self._config = config

    def __call__(
        self,
        state: PlayState,
        target_id: str | None,
        drive_state: DriveState,
    ) -> Move | None:
        observation = _adapt_observation(
            defender_context_observation(
                state,
                self._player_id,
                target_id,
                self._config,
                drive_state,
                self._goal_line,
            ),
            self._expected_size,
        )
        action, _ = self._model.predict(observation, deterministic=True)
        return _move_from_action(action)


def load_policy(path: str | Path, scenario_name: str = "default") -> PolicyQuarterback:
    """Load a saved PPO model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="cpu")
    _ensure_observation_size(model, SUPPORTED_PLAY_SIZES, "qb", path)
    return PolicyQuarterback(model, scenario.config)


def load_drive_policy(path: str | Path, scenario_name: str = "rushed") -> DrivePolicyQuarterback:
    """Load a saved PPO drive model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="cpu")
    _ensure_observation_size(model, SUPPORTED_DRIVE_SIZES, "qb", path)
    return DrivePolicyQuarterback(model, scenario.config)


def load_receiver_policy(
    path: str | Path,
    scenario_name: str = "stage_b",
    player_id: str = "wr1",
) -> ReceiverPolicy:
    """Load a saved PPO receiver model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="cpu")
    _ensure_observation_size(model, SUPPORTED_RECEIVER_SIZES, player_id, path)
    return ReceiverPolicy(model, scenario.config, player_id)


def load_defender_policy(
    path: str | Path,
    scenario_name: str = "stage_b",
    player_id: str = "cb1",
) -> DefenderPolicy:
    """Load a saved PPO defender model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="cpu")
    _ensure_observation_size(model, SUPPORTED_DEFENDER_SIZES, player_id, path)
    return DefenderPolicy(model, scenario.config, player_id)
