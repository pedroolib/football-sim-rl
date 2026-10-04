from math import isfinite

import pytest

from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.evaluation import RuleQuarterbackPolicy
from qb_sim.training.qb_diagnostics import ThrowRecord, run_instrumented_drive
from qb_sim.training.route_audit import aggregate_routes, audit_policy

VALID_ROUTES = {route.value for route in ROUTE_OBSERVATION_ORDER}
STATS_KEYS = {
    "sampled",
    "targeted",
    "completions",
    "yards",
    "interceptions",
    "touchdowns",
    "target_pct",
    "completion_pct",
    "int_pct",
    "td_pct",
    "yards_per_play",
}


def _play(
    routes: tuple[str, ...],
    outcome: str,
    yards: float,
    *,
    touchdown: bool = False,
) -> dict:
    return {
        "down": 1,
        "distance": 10.0,
        "los_before": 40.0,
        "outcome": outcome,
        "spot": 40.0 + yards,
        "yards": yards,
        "first_down": False,
        "touchdown": touchdown,
        "throw_distance": None,
        "sack": outcome == "sack",
        "coverage_sack": False,
        "routes": list(routes),
    }


def _throw(play_index: int, target_id: str, routes: tuple[str, ...]) -> ThrowRecord:
    return ThrowRecord(
        play_index=play_index,
        down=1,
        distance=10.0,
        throw_time=1.0,
        qb_speed=0.0,
        throw_distance=10.0,
        target_id=target_id,
        target_advantage=0.0,
        best_advantage=0.0,
        routes=routes,
    )


def test_aggregate_routes_computes_per_route_counts_and_rates() -> None:
    plays = [
        _play(("go", "slant"), "completed", 12.0),
        _play(("go", "out"), "interception", 0.0),
        _play(("slant", "go"), "touchdown", 20.0, touchdown=True),
    ]
    throws = [
        _throw(1, "wr1", ("go", "slant")),
        _throw(2, "wr2", ("go", "out")),
        _throw(3, "wr1", ("slant", "go")),
    ]

    stats = aggregate_routes(plays, throws)

    assert set(stats) == VALID_ROUTES
    go = stats["go"]
    assert go["sampled"] == 3
    assert go["targeted"] == 1
    assert go["completions"] == 1
    assert go["yards"] == pytest.approx(32.0)
    assert go["interceptions"] == 1
    assert go["touchdowns"] == 1
    assert go["target_pct"] == pytest.approx(100.0 / 3.0)
    assert go["completion_pct"] == pytest.approx(100.0 / 3.0)
    assert go["int_pct"] == pytest.approx(100.0 / 3.0)
    assert go["td_pct"] == pytest.approx(100.0 / 3.0)
    assert go["yards_per_play"] == pytest.approx(32.0 / 3.0)

    slant = stats["slant"]
    assert slant["sampled"] == 2
    assert slant["targeted"] == 1
    assert slant["completions"] == 1
    assert slant["yards"] == pytest.approx(32.0)
    assert slant["interceptions"] == 0
    assert slant["touchdowns"] == 1
    assert slant["target_pct"] == pytest.approx(50.0)
    assert slant["completion_pct"] == pytest.approx(50.0)
    assert slant["int_pct"] == pytest.approx(0.0)
    assert slant["td_pct"] == pytest.approx(50.0)
    assert slant["yards_per_play"] == pytest.approx(16.0)

    out = stats["out"]
    assert out["sampled"] == 1
    assert out["targeted"] == 1
    assert out["completions"] == 0
    assert out["interceptions"] == 1
    assert out["target_pct"] == pytest.approx(100.0)
    assert out["int_pct"] == pytest.approx(100.0)


def test_aggregate_routes_leaves_unsampled_routes_at_zero() -> None:
    stats = aggregate_routes([_play(("go", "slant"), "completed", 12.0)], [])

    for route in ("post", "curl", "flat", "wheel", "corner"):
        assert stats[route]["sampled"] == 0
        assert stats[route]["target_pct"] == pytest.approx(0.0)
        assert stats[route]["yards_per_play"] == pytest.approx(0.0)


def test_aggregate_routes_ignores_throws_with_unknown_targets() -> None:
    plays = [_play(("go", "slant"), "completed", 12.0)]
    throws = [_throw(1, "qb", ("go", "slant"))]

    stats = aggregate_routes(plays, throws)

    assert stats["go"]["targeted"] == 0
    assert stats["slant"]["targeted"] == 0


def test_audit_policy_returns_finite_route_stats_for_the_rule_quarterback() -> None:
    scenario = get_scenario("stage_b")
    policy = RuleQuarterbackPolicy(scenario.config)

    report = audit_policy(policy, "stage_b", (20000, 20001))

    assert set(report) == {"routes", "drives"}
    assert set(report["routes"]) == VALID_ROUTES
    for route_stats in report["routes"].values():
        assert set(route_stats) == STATS_KEYS
        for value in route_stats.values():
            assert isfinite(value)
    assert report["drives"]["count"] == 2
    assert isfinite(report["drives"]["points_per_drive"])
    assert isfinite(report["drives"]["first_downs_per_drive"])


def test_audit_policy_sampled_counts_match_the_recorded_plays() -> None:
    scenario = get_scenario("stage_b")
    policy = RuleQuarterbackPolicy(scenario.config)
    seeds = (20000, 20001)

    report = audit_policy(policy, "stage_b", seeds)

    expected = {route: 0 for route in VALID_ROUTES}
    env = QBDriveEnv(scenario)
    for seed in seeds:
        drive = run_instrumented_drive(env, policy, seed)
        for play in drive.plays:
            for route in play["routes"]:
                expected[route] += 1
    env.close()

    assert report["drives"]["plays"] == sum(expected.values()) // 2
    for route, count in expected.items():
        assert report["routes"][route]["sampled"] == count
