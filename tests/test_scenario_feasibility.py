"""Feasibility guards for the M7 speed scenarios.

The full 100-episode report lives in ``scripts/scenario_feasibility.py``; these
tests use small fixed-seed samples with conservative floors to catch
catastrophic regressions (a scenario where passing becomes impossible or the
rusher arrives before any throw). Sacks are only counted when the engine
reports ``PlayOutcome.SACK``.
"""

import numpy as np
import pytest

from qb_sim.engine.geometry import distance
from qb_sim.rules.scenarios import (
    FAST_COVERAGE_SCENARIO,
    FAST_RUSH_SCENARIO,
    MIXED_SCENARIO,
    RUSHED_SCENARIO,
    SLOW_RUSH_SCENARIO,
)
from qb_sim.training.env import QBPassingEnv
from qb_sim.training.evaluation import RuleQuarterbackPolicy
from qb_sim.training.randomization import RandomizedEnv, SpeedRanges

EPISODES = 10
SEED_START = 10_000


def _run_episode(env, seed: int) -> tuple[str, float | None]:
    observation, _ = env.reset(seed=seed)
    policy = RuleQuarterbackPolicy(env.play.config)
    arrival: float | None = None
    while True:
        state = env.play.state
        if arrival is None:
            for rusher_id in env.play.rusher_ids:
                rusher = state.player(rusher_id)
                if (
                    distance(rusher.position, state.player("qb").position)
                    <= env.play.config.tackle_radius
                ):
                    arrival = state.elapsed_seconds
        observation, _, terminated, truncated, info = env.step(policy(env, observation))
        if terminated or truncated:
            return info["outcome"], arrival


def _stats(env, seed_offset: int = 0) -> tuple[float, float, float | None]:
    rows = [_run_episode(env, seed_offset + SEED_START + index) for index in range(EPISODES)]
    completions = sum(outcome == "completed" for outcome, _ in rows) / EPISODES
    sacks = sum(outcome == "sack" for outcome, _ in rows) / EPISODES
    arrivals = [arrival for _, arrival in rows if arrival is not None]
    return completions, sacks, (min(arrivals) if arrivals else None)


@pytest.mark.parametrize(
    ("scenario", "min_completion", "max_sacks"),
    (
        (RUSHED_SCENARIO, 0.20, 0.15),
        (FAST_COVERAGE_SCENARIO, 0.10, 0.15),
        (SLOW_RUSH_SCENARIO, 0.40, 0.10),
        (FAST_RUSH_SCENARIO, 0.20, 0.15),
        (MIXED_SCENARIO, 0.40, 0.10),
    ),
)
def test_speed_scenarios_keep_passing_viable(
    scenario,
    min_completion: float,
    max_sacks: float,
) -> None:
    env = QBPassingEnv(scenario)
    try:
        completions, sacks, _ = _stats(env)
    finally:
        env.close()

    assert completions >= min_completion
    assert sacks <= max_sacks


def test_fast_rusher_arrives_before_the_slow_one() -> None:
    fast_env = QBPassingEnv(FAST_RUSH_SCENARIO)
    slow_env = QBPassingEnv(SLOW_RUSH_SCENARIO)
    try:
        _, _, fast_arrival = _stats(fast_env)
        _, _, slow_arrival = _stats(slow_env)
    finally:
        fast_env.close()
        slow_env.close()

    assert fast_arrival is not None
    assert fast_arrival >= 0.5
    assert slow_arrival is None or slow_arrival > fast_arrival


def test_randomized_speeds_stay_viable() -> None:
    ranges = SpeedRanges()
    env = RandomizedEnv(RUSHED_SCENARIO, ranges)
    try:
        completions = []
        sacks = []
        arrivals = []
        for config_index in range(4):
            rows = [
                _run_episode(env, config_index * 1_000 + SEED_START + episode)
                for episode in range(3)
            ]
            completions.append(sum(o == "completed" for o, _ in rows) / len(rows))
            sacks.append(sum(o == "sack" for o, _ in rows) / len(rows))
            arrivals.extend(a for _, a in rows if a is not None)
    finally:
        env.close()

    assert min(completions) >= 0.15
    assert max(sacks) <= 0.2
    assert min(arrivals) >= 0.5


def test_rusher_speed_is_applied_per_player() -> None:
    slow = QBPassingEnv(SLOW_RUSH_SCENARIO)
    fast = QBPassingEnv(FAST_RUSH_SCENARIO)
    try:
        slow.reset(seed=0)
        fast.reset(seed=0)
        slow_start = slow.play.state.player("ru1").position.copy()
        fast_start = fast.play.state.player("ru1").position.copy()
        zeros = np.zeros(6, dtype=np.float32)

        slow.step(zeros)
        fast.step(zeros)

        slow_move = distance(slow.play.state.player("ru1").position, slow_start)
        fast_move = distance(fast.play.state.player("ru1").position, fast_start)
        assert slow_move == pytest.approx(5.0 * slow.play.config.dt)
        assert fast_move == pytest.approx(8.0 * fast.play.config.dt)
        assert slow.play.state.player("ru1").max_speed == 5.0
        assert fast.play.state.player("ru1").max_speed == 8.0
    finally:
        slow.close()
        fast.close()
