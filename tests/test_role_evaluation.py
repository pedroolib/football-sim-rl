import numpy as np
import pytest

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.evaluation import RuleQuarterbackPolicy
from qb_sim.training.role_evaluation import (
    evaluate_composition,
    run_composition_episode,
    summarize_composition,
    summarize_role_drive,
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
    assert len(summary["turnover_on_downs_ci"]) == 2
    assert summary["turnover_on_downs_ci"][0] <= 50.0 <= summary["turnover_on_downs_ci"][1]
    assert len(summary["sack_ci"]) == 2


def test_summarize_role_drive_reports_touchdown_interval() -> None:
    plays = [
        {
            "outcome": "sack",
            "pressure_at_throw": None,
            "arrival_time": 1.2,
            "separation_allowed_throw": None,
        },
    ]
    results = [{"outcome": "touchdown", "points": 7.0, "plays": plays}]

    summary = summarize_role_drive(results)

    assert summary["touchdown_pct_allowed"] == pytest.approx(100.0)
    assert summary["touchdown_pct_allowed_ci"][0] <= 100.0
    assert summary["touchdown_pct_allowed_ci"][1] == pytest.approx(100.0)


def test_summarize_role_drive_reports_completion_and_interception_rates() -> None:
    plays = [
        {
            "outcome": "completed",
            "pressure_at_throw": 0.3,
            "arrival_time": None,
            "separation_allowed_throw": 1.5,
        },
        {
            "outcome": "interception",
            "pressure_at_throw": None,
            "arrival_time": None,
            "separation_allowed_throw": None,
        },
    ]
    results = [{"outcome": "interception", "points": 0.0, "plays": plays}]

    summary = summarize_role_drive(results)

    assert summary["completion_allowed_pct"] == pytest.approx(50.0)
    assert summary["completion_allowed_ci"][0] <= 50.0 <= summary["completion_allowed_ci"][1]
    assert summary["interception_forced_pct"] == pytest.approx(50.0)
    assert summary["interception_forced_ci"][0] <= 50.0 <= summary["interception_forced_ci"][1]


def test_run_composition_episode_records_pre_step_pressure_at_throw() -> None:
    from qb_sim.training.drive_env import QBDriveEnv

    scenario = get_scenario("stage_b")
    throw_action = np.zeros(6, dtype=np.float32)
    throw_action[5] = 1.0
    observed: dict[str, object] = {"throwing": False, "pressure": None}

    def throwing_policy(env, observation) -> np.ndarray:
        pressure = float(env.play.state.pressure)
        if not observed["throwing"] and pressure > 0.0:
            observed["pressure"] = pressure
            observed["throwing"] = True
        if observed["throwing"]:
            return throw_action
        move = np.zeros(6, dtype=np.float32)
        move[0] = 1.0
        return move

    env = QBDriveEnv(scenario)
    result = run_composition_episode(env, throwing_policy, 0)
    env.close()

    assert observed["pressure"] is not None
    assert result["pressure_at_throw"] == pytest.approx(observed["pressure"])


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
