"""Role-level and composition metrics for Stage B over fixed seed sets."""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.drive_evaluation import wilson_interval
from qb_sim.training.role_env import RoleDriveEnv, RolePlayEnv

QBPolicy = Callable[[QBDriveEnv, np.ndarray], np.ndarray]


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _ci(count: int, total: int) -> list[float]:
    low, high = wilson_interval(count, total)
    return [100.0 * low, 100.0 * high]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def run_role_play_episode(env: RolePlayEnv, action_fn, seed: int) -> dict[str, Any]:
    """Run one seeded play with a learner policy and collect receiver metrics."""
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = np.asarray(action_fn(env, observation), dtype=np.float32)
        observation, _, terminated, truncated, info = env.step(action)
    return dict(info)


def evaluate_role_play(action_fn, scenario_name: str, seeds: Iterable[int], **kwargs) -> dict:
    env = RolePlayEnv(get_scenario(scenario_name), **kwargs)
    results = [run_role_play_episode(env, action_fn, seed) for seed in seeds]
    env.close()
    return summarize_role_play(results)


def summarize_role_play(results: list[dict[str, Any]]) -> dict:
    total = len(results)
    if total == 0:
        raise ValueError("at least one play is required")
    catches = sum(bool(result["caught"]) for result in results)
    touchdowns = sum(result["outcome"] == "touchdown" for result in results)
    separations = [
        float(result["separation_at_throw"])
        for result in results
        if result["separation_at_throw"] is not None
    ]
    return {
        "episodes": float(total),
        "catch_pct": _pct(catches, total),
        "catch_ci": _ci(catches, total),
        "touchdown_pct": _pct(touchdowns, total),
        "touchdown_ci": _ci(touchdowns, total),
        "yards_per_play": _mean([float(result["yards"]) for result in results]),
        "mean_separation_at_throw": _mean(separations),
    }


def run_role_drive_episode(env: RoleDriveEnv, action_fn, seed: int) -> dict[str, Any]:
    """Run one seeded drive with a learner policy and collect defender metrics."""
    env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        observation = env.observation()
        action = np.asarray(action_fn(env, observation), dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)
    state = env.drive.state
    return {
        "outcome": state.end_reason or "truncated",
        "points": state.points,
        "plays": list(env.play_history),
    }


def evaluate_role_drive(action_fn, scenario_name: str, seeds: Iterable[int], **kwargs) -> dict:
    env = RoleDriveEnv(get_scenario(scenario_name), **kwargs)
    results = [run_role_drive_episode(env, action_fn, seed) for seed in seeds]
    env.close()
    return summarize_role_drive(results)


def summarize_role_drive(results: list[dict[str, Any]]) -> dict:
    total = len(results)
    if total == 0:
        raise ValueError("at least one drive is required")
    plays = [play for result in results for play in result["plays"]]
    sacks = sum(play["outcome"] == "sack" for play in plays)
    pressures = [
        float(play["pressure_at_throw"])
        for play in plays
        if play["pressure_at_throw"] is not None
    ]
    arrivals = [float(play["arrival_time"]) for play in plays if play["arrival_time"] is not None]
    separations = [
        float(play["separation_allowed_throw"])
        for play in plays
        if play["separation_allowed_throw"] is not None
    ]
    touchdowns = sum(result["outcome"] == "touchdown" for result in results)
    return {
        "episodes": float(total),
        "touchdown_pct_allowed": _pct(touchdowns, total),
        "touchdown_pct_allowed_ci": _ci(touchdowns, total),
        "sack_pct": _pct(sacks, len(plays)),
        "sack_ci": _ci(sacks, len(plays)),
        "mean_pressure_at_throw": _mean(pressures),
        "mean_arrival_time": _mean(arrivals),
        "mean_separation_allowed_throw": _mean(separations),
        "plays_per_drive": len(plays) / total,
    }


def run_composition_episode(env: QBDriveEnv, qb_policy: QBPolicy, seed: int) -> dict[str, Any]:
    """Run one seeded drive with the QB policy and frozen role opponents."""
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    separation_throw: float | None = None
    separation_catch: float | None = None
    pressure_throw: float | None = None
    arrival: float | None = None
    while not (terminated or truncated):
        pre_pressure = float(env.play.state.pressure)
        action = np.asarray(qb_policy(env, observation), dtype=np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        current = env.play.state
        if separation_throw is None and current.throw_time is not None:
            separation_throw = _target_separation(env)
            pressure_throw = pre_pressure
        if separation_catch is None and current.carrier_id is not None:
            separation_catch = _target_separation(env)
        if arrival is None and env.play.rusher_ids:
            rusher = current.player(env.play.rusher_ids[0])
            quarterback = current.player("qb")
            if (
                float(np.linalg.norm(rusher.position - quarterback.position))
                <= env.play.config.tackle_radius
            ):
                arrival = float(current.elapsed_seconds)
    state = env.drive.state
    return {
        "outcome": state.end_reason or "truncated",
        "points": state.points,
        "first_downs": state.first_downs,
        "plays": list(env.play_history),
        "separation_at_throw": separation_throw,
        "separation_at_catch": separation_catch,
        "pressure_at_throw": pressure_throw,
        "arrival_time": arrival,
    }


def _target_separation(env: QBDriveEnv) -> float | None:
    state = env.play.state
    if state.carrier_id is not None:
        receiver_id = state.carrier_id
    else:
        target = env.play.state.ball.target
        if target is None:
            return None
        receiver_id = min(
            env.play.receiver_ids,
            key=lambda player_id: float(np.linalg.norm(state.player(player_id).position - target)),
        )
    receiver = state.player(receiver_id)
    defenders = [state.player(player_id).position for player_id in env.play.defender_ids]
    if not defenders:
        return None
    return float(min(np.linalg.norm(receiver.position - defender) for defender in defenders))


def evaluate_composition(
    qb_policy: QBPolicy,
    opponents: dict,
    scenario_name: str,
    seeds: Iterable[int],
) -> dict:
    env = QBDriveEnv(get_scenario(scenario_name), opponents=opponents)
    results = [run_composition_episode(env, qb_policy, seed) for seed in seeds]
    env.close()
    return summarize_composition(results)


def summarize_composition(results: list[dict[str, Any]]) -> dict:
    total = len(results)
    if total == 0:
        raise ValueError("at least one drive is required")
    touchdowns = sum(result["outcome"] == "touchdown" for result in results)
    interceptions = sum(result["outcome"] == "interception" for result in results)
    turnovers = sum(result["outcome"] == "turnover_on_downs" for result in results)
    plays = [play for result in results for play in result["plays"]]
    third_downs = [play for play in plays if play["down"] == 3]
    conversions = sum(bool(play["first_down"] or play["touchdown"]) for play in third_downs)
    sacks = sum(play["outcome"] == "sack" for play in plays)
    separations = [
        float(result["separation_at_throw"])
        for result in results
        if result["separation_at_throw"] is not None
    ]
    pressures = [
        float(result["pressure_at_throw"])
        for result in results
        if result["pressure_at_throw"] is not None
    ]
    arrivals = [
        float(result["arrival_time"])
        for result in results
        if result["arrival_time"] is not None
    ]
    return {
        "episodes": float(total),
        "touchdown_pct": _pct(touchdowns, total),
        "touchdown_ci": _ci(touchdowns, total),
        "points_per_drive": _mean([float(result["points"]) for result in results]),
        "first_downs_per_drive": _mean([float(result["first_downs"]) for result in results]),
        "plays_per_drive": len(plays) / total,
        "third_down_conversion_pct": _pct(conversions, len(third_downs)),
        "third_down_conversion_ci": _ci(conversions, len(third_downs)),
        "interception_pct": _pct(interceptions, total),
        "interception_ci": _ci(interceptions, total),
        "turnover_on_downs_pct": _pct(turnovers, total),
        "turnover_on_downs_ci": _ci(turnovers, total),
        "sack_pct": _pct(sacks, len(plays)),
        "sack_ci": _ci(sacks, len(plays)),
        "mean_separation_at_throw": _mean(separations),
        "mean_pressure_at_throw": _mean(pressures),
        "mean_arrival_time": _mean(arrivals),
    }
