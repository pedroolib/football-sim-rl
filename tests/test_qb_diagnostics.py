from math import isfinite

import pytest

from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.evaluation import RuleQuarterbackPolicy
from qb_sim.training.qb_diagnostics import (
    InstrumentedDrive,
    ThrowRecord,
    run_instrumented_drive,
    summarize_diagnostics,
)

EXPECTED_SUMMARY_KEYS = {
    "drives",
    "touchdown_pct",
    "interception_pct",
    "turnover_on_downs_pct",
    "points_per_drive",
    "first_downs_per_drive",
    "plays_per_drive",
    "sack_pct",
    "coverage_sack_pct",
    "incomplete_pct",
    "throw_rate_pct",
    "mean_throw_time",
    "median_throw_time",
    "moving_throw_pct",
    "mean_qb_speed",
    "mean_throw_distance",
    "mean_best_advantage",
    "mean_target_advantage",
    "decision_gap",
    "pct_target_max_advantage",
    "pct_deep_open_throws",
    "depth_advantage_corr",
}


def _play(
    outcome: str,
    *,
    sack: bool = False,
    coverage_sack: bool = False,
    throw_distance: float | None = None,
) -> dict:
    return {
        "down": 1,
        "distance": 10.0,
        "los_before": 40.0,
        "outcome": outcome,
        "spot": 45.0,
        "yards": 5.0,
        "first_down": False,
        "touchdown": False,
        "throw_distance": throw_distance,
        "sack": sack,
        "coverage_sack": coverage_sack,
    }


def _record(
    *,
    play_index: int = 1,
    down: int = 1,
    distance: float = 10.0,
    throw_time: float = 1.0,
    qb_speed: float = 0.0,
    throw_distance: float = 10.0,
    target_id: str = "wr1",
    target_advantage: float = 0.0,
    best_advantage: float = 0.0,
    routes: tuple[str, ...] = ("go", "slant"),
) -> ThrowRecord:
    return ThrowRecord(
        play_index=play_index,
        down=down,
        distance=distance,
        throw_time=throw_time,
        qb_speed=qb_speed,
        throw_distance=throw_distance,
        target_id=target_id,
        target_advantage=target_advantage,
        best_advantage=best_advantage,
        routes=routes,
    )


def _drive(
    *,
    outcome: str = "touchdown",
    points: float = 7.0,
    first_downs: int = 1,
    records: list[ThrowRecord] | None = None,
    plays: list[dict] | None = None,
) -> InstrumentedDrive:
    return InstrumentedDrive(
        outcome=outcome,
        points=points,
        first_downs=first_downs,
        records=tuple(records or []),
        plays=tuple(plays or []),
    )


def test_summarize_diagnostics_aggregates_drive_and_play_outcomes() -> None:
    drives = [
        _drive(
            outcome="touchdown",
            points=7.0,
            first_downs=2,
            records=[_record()],
            plays=[
                _play("completed", throw_distance=10.0),
                _play("incomplete", throw_distance=20.0),
                _play("sack", sack=True),
            ],
        ),
        _drive(
            outcome="turnover_on_downs",
            points=0.0,
            first_downs=0,
            records=[],
            plays=[_play("sack", sack=True, coverage_sack=True)],
        ),
    ]

    summary = summarize_diagnostics(drives)

    assert summary["drives"] == 2.0
    assert summary["touchdown_pct"] == pytest.approx(50.0)
    assert summary["interception_pct"] == pytest.approx(0.0)
    assert summary["turnover_on_downs_pct"] == pytest.approx(50.0)
    assert summary["points_per_drive"] == pytest.approx(3.5)
    assert summary["first_downs_per_drive"] == pytest.approx(1.0)
    assert summary["plays_per_drive"] == pytest.approx(2.0)
    assert summary["throw_rate_pct"] == pytest.approx(50.0)
    assert summary["sack_pct"] == pytest.approx(50.0)
    assert summary["coverage_sack_pct"] == pytest.approx(25.0)
    assert summary["incomplete_pct"] == pytest.approx(25.0)


def test_summarize_diagnostics_measures_decision_gap_and_targeting() -> None:
    records = [
        _record(
            qb_speed=1.0,
            throw_distance=5.0,
            target_advantage=0.5,
            best_advantage=1.0,
            throw_time=0.8,
        ),
        _record(
            qb_speed=0.5,
            throw_distance=10.0,
            target_advantage=0.5,
            best_advantage=0.5,
            throw_time=1.2,
        ),
        _record(
            qb_speed=0.6,
            throw_distance=15.0,
            target_advantage=0.5,
            best_advantage=0.5,
            throw_time=1.6,
        ),
    ]

    summary = summarize_diagnostics([_drive(records=records)])

    assert summary["decision_gap"] == pytest.approx((0.5 + 0.0 + 0.0) / 3.0)
    assert summary["pct_target_max_advantage"] == pytest.approx(100.0 * 2.0 / 3.0)
    assert summary["moving_throw_pct"] == pytest.approx(100.0 * 2.0 / 3.0)
    assert summary["mean_qb_speed"] == pytest.approx(0.7)
    assert summary["mean_throw_distance"] == pytest.approx(10.0)
    assert summary["mean_best_advantage"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_target_advantage"] == pytest.approx(0.5)
    assert summary["mean_throw_time"] == pytest.approx(1.2)
    assert summary["median_throw_time"] == pytest.approx(1.2)
    assert summary["pct_deep_open_throws"] == pytest.approx(100.0 / 3.0)


def test_summarize_diagnostics_correlates_depth_with_best_advantage() -> None:
    records = [
        _record(throw_distance=depth, best_advantage=advantage)
        for depth, advantage in zip((5.0, 10.0, 15.0), (0.1, 0.2, 0.3), strict=True)
    ]

    summary = summarize_diagnostics([_drive(records=records)])

    assert summary["depth_advantage_corr"] == pytest.approx(1.0)


def test_summarize_diagnostics_detects_negative_depth_advantage_correlation() -> None:
    records = [
        _record(throw_distance=depth, best_advantage=advantage)
        for depth, advantage in zip((5.0, 10.0, 15.0), (0.3, 0.2, 0.1), strict=True)
    ]

    summary = summarize_diagnostics([_drive(records=records)])

    assert summary["depth_advantage_corr"] == pytest.approx(-1.0)


def test_summarize_diagnostics_correlation_rejects_small_or_constant_samples() -> None:
    too_few = [
        _record(throw_distance=5.0, best_advantage=0.1),
        _record(throw_distance=15.0, best_advantage=0.3),
    ]
    constant = [
        _record(throw_distance=depth, best_advantage=0.5) for depth in (5.0, 10.0, 15.0)
    ]

    assert summarize_diagnostics([_drive(records=too_few)])["depth_advantage_corr"] is None
    assert summarize_diagnostics([_drive(records=constant)])["depth_advantage_corr"] is None


def test_run_instrumented_drive_with_rule_quarterback_records_throws() -> None:
    scenario = get_scenario("stage_b")
    env = QBDriveEnv(scenario)
    policy = RuleQuarterbackPolicy(scenario.config)

    drives = [run_instrumented_drive(env, policy, seed) for seed in (20000, 20001)]
    env.close()

    assert len(drives) == 2
    assert all(drive.plays for drive in drives)
    records = [record for drive in drives for record in drive.records]
    assert records
    valid_routes = {route.value for route in ROUTE_OBSERVATION_ORDER}
    for record in records:
        assert record.target_id in ("wr1", "wr2")
        assert record.play_index >= 1
        assert isfinite(record.throw_time)
        assert isfinite(record.qb_speed)
        assert isfinite(record.throw_distance)
        assert isfinite(record.target_advantage)
        assert isfinite(record.best_advantage)
        assert len(record.routes) == 2
        assert set(record.routes) <= valid_routes

    summary = summarize_diagnostics(drives)

    assert EXPECTED_SUMMARY_KEYS <= set(summary)
