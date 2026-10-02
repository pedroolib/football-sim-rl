import numpy as np
import pytest

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.evaluation import RuleQuarterbackPolicy
from qb_sim.training.role_evaluation import (
    evaluate_composition,
    summarize_composition,
)


def test_summarize_composition_computes_team_metrics() -> None:
    results = [
        {
            "outcome": "touchdown",
            "points": 7.0,
            "first_downs": 2,
            "plays": [
                {"down": 3, "distance": 12.0, "first_down": True, "touchdown": False,
                 "throw_distance": 18.0, "outcome": "completed", "coverage_sack": False},
            ],
            "separation_at_throw": 3.0,
            "separation_at_catch": 1.0,
            "pressure_at_throw": 0.4,
            "arrival_time": 1.6,
        },
        {
            "outcome": "turnover_on_downs",
            "points": 0.0,
            "first_downs": 0,
            "plays": [
                {"down": 3, "distance": 5.0, "first_down": False, "touchdown": False,
                 "throw_distance": 9.0, "outcome": "incomplete", "coverage_sack": False},
            ],
            "separation_at_throw": 2.0,
            "separation_at_catch": None,
            "pressure_at_throw": None,
            "arrival_time": None,
        },
    ]

    summary = summarize_composition(results)

    assert summary["episodes"] == 2.0
    assert summary["touchdown_pct"] == pytest.approx(50.0)
    assert summary["points_per_drive"] == pytest.approx(3.5)
    assert summary["third_down_conversion_pct"] == pytest.approx(50.0)
    assert summary["mean_separation_at_throw"] == pytest.approx(2.5)


def test_evaluate_composition_with_the_rule_quarterback_is_reproducible() -> None:
    scenario = get_scenario("stage_b")
    policy = RuleQuarterbackPolicy(scenario.config)

    first = evaluate_composition(policy, {}, "stage_b", range(2))
    second = evaluate_composition(policy, {}, "stage_b", range(2))

    assert first == second


def test_qb_drive_env_calls_context_aware_opponents() -> None:
    from qb_sim.training.drive_env import QBDriveEnv

    calls: list[object] = []

    def opponent(state, route):
        calls.append(route)
        return None

    env = QBDriveEnv(get_scenario("stage_b"), opponents={"wr1": opponent})
    env.reset(seed=0)
    env.step(np.zeros(6, dtype=np.float32))

    assert calls
