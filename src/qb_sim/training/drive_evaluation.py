"""Aggregate metrics for drive episodes over fixed seed sets."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import sqrt

import numpy as np

from qb_sim.rules.drive import DriveConfig
from qb_sim.training.drive_env import QBDriveEnv

ActionFn = Callable[[QBDriveEnv, np.ndarray], np.ndarray]
THIRD_AND_LONG_YARDS = 8.0
CHECKDOWN_YARDS = 10.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True, slots=True)
class DriveEpisodeResult:
    """Per-drive diagnostics used by the aggregate metrics."""

    outcome: str
    points: float
    first_downs: int
    plays: tuple[dict, ...]


def run_drive_episode(
    env: QBDriveEnv,
    action_fn: ActionFn,
    seed: int,
) -> DriveEpisodeResult:
    """Run one seeded drive and collect its play history."""
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = np.asarray(action_fn(env, observation), dtype=np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
    state = env.drive.state
    return DriveEpisodeResult(
        outcome=state.end_reason or "truncated",
        points=state.points,
        first_downs=state.first_downs,
        plays=tuple(env.play_history),
    )


def evaluate_drive(
    action_fn: ActionFn,
    scenario_name: str,
    seeds: Iterable[int],
    drive_config: DriveConfig | None = None,
) -> dict:
    """Evaluate a policy on fixed drive seeds and aggregate its metrics."""
    from qb_sim.rules.scenarios import get_scenario

    env = QBDriveEnv(get_scenario(scenario_name), drive_config=drive_config)
    results = [run_drive_episode(env, action_fn, seed) for seed in seeds]
    env.close()
    return summarize_drives(results)


def summarize_drives(results: list[DriveEpisodeResult]) -> dict:
    """Aggregate drive results into strategy, scoring, and efficiency metrics."""
    total = len(results)
    if total == 0:
        raise ValueError("at least one drive is required")

    def pct(count: int, denominator: int) -> float:
        return 100.0 * count / denominator if denominator else 0.0

    def ci(count: int, denominator: int) -> list[float]:
        low, high = wilson_interval(count, denominator)
        return [100.0 * low, 100.0 * high]

    touchdowns = sum(result.outcome == "touchdown" for result in results)
    interceptions = sum(result.outcome == "interception" for result in results)
    turnovers = sum(result.outcome == "turnover_on_downs" for result in results)
    plays = [play for result in results for play in result.plays]
    third_downs = [play for play in plays if play["down"] == 3]
    third_and_long = [
        play for play in third_downs if play["distance"] >= THIRD_AND_LONG_YARDS
    ]
    third_conversions = sum(
        bool(play["first_down"] or play["touchdown"]) for play in third_downs
    )
    long_conversions = sum(
        bool(play["first_down"] or play["touchdown"]) for play in third_and_long
    )
    long_throws = [
        float(play["throw_distance"])
        for play in third_and_long
        if play["throw_distance"] is not None
    ]
    checkdowns = sum(distance <= CHECKDOWN_YARDS for distance in long_throws)
    sacks = sum(play["outcome"] == "sack" for play in plays)
    coverage_sacks = sum(bool(play["coverage_sack"]) for play in plays)
    incompletes = sum(play["outcome"] == "incomplete" for play in plays)

    return {
        "episodes": float(total),
        "touchdown_pct": pct(touchdowns, total),
        "touchdown_ci": ci(touchdowns, total),
        "interception_pct": pct(interceptions, total),
        "interception_ci": ci(interceptions, total),
        "turnover_on_downs_pct": pct(turnovers, total),
        "turnover_on_downs_ci": ci(turnovers, total),
        "points_per_drive": sum(result.points for result in results) / total,
        "first_downs_per_drive": sum(result.first_downs for result in results) / total,
        "plays_per_drive": len(plays) / total,
        "third_down_conversion_pct": pct(third_conversions, len(third_downs)),
        "third_down_conversion_ci": ci(third_conversions, len(third_downs)),
        "third_and_long_conversion_pct": pct(long_conversions, len(third_and_long)),
        "third_and_long_conversion_ci": ci(long_conversions, len(third_and_long)),
        "third_and_long_throw_depth_yards": (
            sum(long_throws) / len(long_throws) if long_throws else 0.0
        ),
        "checkdown_pct": pct(checkdowns, len(long_throws)),
        "checkdown_ci": ci(checkdowns, len(long_throws)),
        "sack_pct": pct(sacks, len(plays)),
        "sack_ci": ci(sacks, len(plays)),
        "coverage_sack_pct": pct(coverage_sacks, len(plays)),
        "coverage_sack_ci": ci(coverage_sacks, len(plays)),
        "incomplete_pct": pct(incompletes, len(plays)),
        "incomplete_ci": ci(incompletes, len(plays)),
    }
