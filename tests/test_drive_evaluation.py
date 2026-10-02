import pytest

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.drive_evaluation import (
    DriveEpisodeResult,
    evaluate_drive,
    run_drive_episode,
    summarize_drives,
    wilson_interval,
)
from qb_sim.training.evaluation import RuleQuarterbackPolicy


def _play(down: int, distance: float, outcome: str, **overrides) -> dict:
    play = {
        "down": down,
        "distance": distance,
        "los_before": 40.0,
        "outcome": outcome,
        "spot": 45.0,
        "yards": 5.0,
        "first_down": False,
        "touchdown": False,
        "throw_distance": 12.0,
        "sack": outcome == "sack",
        "coverage_sack": False,
    }
    play.update(overrides)
    return play


def _result(outcome: str, plays: list[dict], points: float = 0.0) -> DriveEpisodeResult:
    return DriveEpisodeResult(
        outcome=outcome,
        points=points,
        first_downs=sum(1 for play in plays if play["first_down"]),
        plays=tuple(plays),
    )


def test_wilson_interval_brackets_the_sample_proportion() -> None:
    low, high = wilson_interval(50, 100)

    assert low < 0.5 < high
    assert 0.0 <= low < high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_summarize_drives_computes_third_down_and_depth_metrics() -> None:
    third_and_long = _play(3, 15.0, "incomplete", throw_distance=25.0)
    third_and_short = _play(3, 2.0, "completed", first_down=True, throw_distance=4.0)
    conversion = _play(1, 10.0, "completed", first_down=True)
    results = [
        _result("touchdown", [conversion, third_and_long], points=7.0),
        _result("turnover_on_downs", [third_and_short], points=0.0),
    ]

    summary = summarize_drives(results)

    assert summary["episodes"] == 2.0
    assert summary["touchdown_pct"] == pytest.approx(50.0)
    assert summary["turnover_on_downs_pct"] == pytest.approx(50.0)
    assert summary["third_down_conversion_pct"] == pytest.approx(50.0)
    assert summary["third_and_long_conversion_pct"] == pytest.approx(0.0)
    assert summary["third_and_long_throw_depth_yards"] == pytest.approx(25.0)
    assert summary["checkdown_pct"] == pytest.approx(0.0)
    assert summary["sack_pct"] == pytest.approx(0.0)
    assert summary["sack_ci"][0] == pytest.approx(0.0)
    assert summary["sack_ci"][1] > 0.0
    assert summary["coverage_sack_ci"] == summary["sack_ci"]
    assert summary["incomplete_ci"] != summary["sack_ci"]
    assert summary["incomplete_ci"][0] > 0.0
    assert summary["incomplete_ci"][1] > summary["sack_ci"][1]
    assert summary["plays_per_drive"] == pytest.approx(1.5)


def test_summarize_drives_handles_missing_third_downs() -> None:
    results = [_result("max_plays", [_play(1, 10.0, "completed", first_down=True)])]

    summary = summarize_drives(results)

    assert summary["third_down_conversion_pct"] == 0.0
    assert summary["third_down_conversion_ci"] == [0.0, 0.0]
    assert summary["third_and_long_conversion_pct"] == 0.0
    assert summary["checkdown_pct"] == 0.0


def test_evaluate_drive_with_the_rule_quarterback_is_reproducible() -> None:
    scenario = get_scenario("rushed")
    policy = RuleQuarterbackPolicy(scenario.config)

    first = evaluate_drive(policy, "rushed", range(3))
    second = evaluate_drive(policy, "rushed", range(3))

    assert first == second
    assert first["episodes"] == 3.0


def test_run_drive_episode_returns_the_recorded_plays() -> None:
    scenario = get_scenario("rushed")
    env = QBDriveEnv(scenario)
    result = run_drive_episode(env, RuleQuarterbackPolicy(scenario.config), seed=0)

    assert result.outcome in ("touchdown", "interception", "turnover_on_downs", "max_plays")
    assert len(result.plays) >= 1
    env.close()
