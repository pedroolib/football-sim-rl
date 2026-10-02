"""Deterministic policy evaluation over fixed seed sets."""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qb_sim.agents.rule_controllers import HeuristicQuarterback
from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import BallStatus
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.actions import ACTION_SIZE, encode_engine_action
from qb_sim.training.env import QBPassingEnv

ActionFn = Callable[[QBPassingEnv, np.ndarray], np.ndarray]
_OUTCOMES = ("completed", "interception", "incomplete", "out_of_bounds", "time_limit")


class RandomPolicy:
    """Uniform random policy with its own deterministic generator."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, env: QBPassingEnv, observation: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=ACTION_SIZE).astype(np.float32)


class RuleQuarterbackPolicy:
    """The M3 heuristic quarterback, encoded as environment actions."""

    def __init__(self, config: SimulationConfig) -> None:
        self._agent = HeuristicQuarterback("qb", config)

    def __call__(self, env: QBPassingEnv, observation: np.ndarray) -> np.ndarray:
        action = self._agent.decide(env.play.state)
        if action is None:
            return np.zeros(ACTION_SIZE, dtype=np.float32)
        return encode_engine_action(action)


class PPOActionPolicy:
    """Evaluate a saved PPO model through the environment action API."""

    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        from stable_baselines3 import PPO

        self._model = PPO.load(str(model_path), device=device)

    def __call__(self, env: QBPassingEnv, observation: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Per-episode diagnostics used by the aggregate metrics."""

    outcome: str
    reward: float
    duration: float
    yards: float
    threw: bool
    throw_time: float | None
    move_fraction: float
    pressure_at_throw: float | None
    qb_speed_at_throw: float | None


def run_episode(env: QBPassingEnv, action_fn: ActionFn, seed: int) -> EpisodeResult:
    """Run one seeded episode and collect its metrics."""
    observation, info = env.reset(seed=seed)
    scrimmage = env.scenario.config.quarterback_start[0]
    total_reward = 0.0
    move_ticks = 0
    ticks = 0
    pressure_at_throw: float | None = None
    qb_speed_at_throw: float | None = None
    terminated = truncated = False

    while not (terminated or truncated):
        action = np.asarray(action_fn(env, observation), dtype=np.float32)
        state = env.play.state
        intends_throw = state.ball.status is BallStatus.HELD and float(action[5]) > 0.0
        if intends_throw:
            if pressure_at_throw is None:
                pressure_at_throw = state.pressure
                qb_speed_at_throw = float(np.linalg.norm(state.player("qb").velocity))
        elif state.ball.status is BallStatus.HELD and float(np.linalg.norm(action[:2])) > 0.1:
            move_ticks += 1
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        ticks += 1

    yards = 0.0
    if info["carrier_id"] is not None:
        carrier = env.play.state.player(info["carrier_id"])
        yards = float(carrier.position[0] - scrimmage)
    return EpisodeResult(
        outcome=info["outcome"],
        reward=total_reward,
        duration=float(info["elapsed_seconds"]),
        yards=yards,
        threw=info["throw_time"] is not None,
        throw_time=info["throw_time"],
        move_fraction=(move_ticks / ticks) if ticks else 0.0,
        pressure_at_throw=pressure_at_throw,
        qb_speed_at_throw=qb_speed_at_throw,
    )


def evaluate(
    action_fn: ActionFn,
    scenario_name: str,
    seeds: Iterable[int],
) -> dict[str, float]:
    """Evaluate a policy on a fixed seed set and aggregate its metrics."""
    env = QBPassingEnv(get_scenario(scenario_name))
    results = [run_episode(env, action_fn, seed) for seed in seeds]
    env.close()
    return summarize_episodes(results)


def summarize_episodes(results: list[EpisodeResult]) -> dict[str, float]:
    """Aggregate episode results into the comparison metrics."""
    total = len(results)
    if total == 0:
        raise ValueError("at least one episode is required")

    counts = Counter(result.outcome for result in results)
    throws = [result for result in results if result.throw_time is not None]
    pressures = [
        result.pressure_at_throw for result in throws if result.pressure_at_throw is not None
    ]
    speeds = [result.qb_speed_at_throw for result in throws if result.qb_speed_at_throw is not None]
    summary: dict[str, float] = {
        "episodes": float(total),
        "yards_per_attempt": sum(result.yards for result in results) / total,
        "mean_reward": sum(result.reward for result in results) / total,
        "average_duration": sum(result.duration for result in results) / total,
        "throw_frequency": 100.0 * len(throws) / total,
        "move_frequency": 100.0 * sum(result.move_fraction for result in results) / total,
        "average_throw_time": (
            sum(result.throw_time for result in throws) / len(throws) if throws else 0.0
        ),
        "average_pressure_at_throw": (sum(pressures) / len(pressures) if pressures else 0.0),
        "average_qb_speed_at_throw": (sum(speeds) / len(speeds) if speeds else 0.0),
    }
    for outcome in _OUTCOMES:
        summary[f"{outcome}_pct"] = 100.0 * counts.get(outcome, 0) / total
    return summary
