"""Data-driven route audit for drive quarterbacks (Fase 0, E0.4).

Pure aggregation on top of :mod:`qb_sim.training.qb_diagnostics`: it joins the
throws of a drive to its plays by ``play_index`` (the 1-based play number) and
attributes each play to every route sampled on it, so a route is judged by the
outcome of the plays it was part of.

Rates are percentages in ``[0, 100]`` and use the play association as
denominator: ``target_pct`` is the share of sampled plays where the route was
the target, while ``completion_pct``, ``int_pct`` and ``td_pct`` are the shares
of sampled plays that ended in a completion, interception, or touchdown.
"""

from collections.abc import Iterable
from typing import Any

from qb_sim.rules.drive import DriveConfig
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.qb_diagnostics import ThrowRecord, run_instrumented_drive


def _empty_counts() -> dict[str, dict[str, float | int]]:
    return {
        route.value: {
            "sampled": 0,
            "targeted": 0,
            "completions": 0,
            "yards": 0.0,
            "interceptions": 0,
            "touchdowns": 0,
        }
        for route in ROUTE_OBSERVATION_ORDER
    }


def _plays_by_route(plays: list[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {route.value: [] for route in ROUTE_OBSERVATION_ORDER}
    for play in plays:
        for route in play.get("routes", ()):
            if route in index:
                index[route].append(play)
    return index


def _target_route(record: ThrowRecord) -> str | None:
    """Route run by the targeted receiver (``wrN`` follows the route order)."""
    if not record.target_id.startswith("wr"):
        return None
    try:
        index = int(record.target_id[2:]) - 1
    except ValueError:
        return None
    if 0 <= index < len(record.routes):
        return record.routes[index]
    return None


def _count_routes(plays: list[dict], throws: list[ThrowRecord]) -> dict[str, dict]:
    counts = _empty_counts()
    by_route = _plays_by_route(plays)
    throws_by_play: dict[int, list[ThrowRecord]] = {}
    for record in throws:
        throws_by_play.setdefault(record.play_index, []).append(record)

    for route, route_plays in by_route.items():
        stats = counts[route]
        stats["sampled"] = len(route_plays)
        stats["yards"] = sum(float(play["yards"]) for play in route_plays)
        stats["completions"] = sum(play["outcome"] == "completed" for play in route_plays)
        stats["interceptions"] = sum(
            play["outcome"] == "interception" for play in route_plays
        )
        stats["touchdowns"] = sum(bool(play["touchdown"]) for play in route_plays)

    for position, _ in enumerate(plays, start=1):
        for record in throws_by_play.get(position, ()):
            route = _target_route(record)
            if route in counts:
                counts[route]["targeted"] += 1
    return counts


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def _rates(counts: dict[str, dict]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for route, raw in counts.items():
        sampled = int(raw["sampled"])
        targeted = int(raw["targeted"])
        result[route] = {
            "sampled": sampled,
            "targeted": targeted,
            "completions": int(raw["completions"]),
            "yards": float(raw["yards"]),
            "interceptions": int(raw["interceptions"]),
            "touchdowns": int(raw["touchdowns"]),
            "target_pct": _pct(targeted, sampled),
            "completion_pct": _pct(raw["completions"], sampled),
            "int_pct": _pct(raw["interceptions"], sampled),
            "td_pct": _pct(raw["touchdowns"], sampled),
            "yards_per_play": raw["yards"] / sampled if sampled else 0.0,
        }
    return result


def _merge_counts(
    totals: dict[str, dict], counts: dict[str, dict]
) -> None:
    for route, stats in counts.items():
        target = totals[route]
        for key in ("sampled", "targeted", "completions", "interceptions", "touchdowns"):
            target[key] = int(target[key]) + int(stats[key])
        target["yards"] = float(target["yards"]) + float(stats["yards"])


def aggregate_routes(plays: list[dict], throws: list[ThrowRecord]) -> dict[str, dict]:
    """Aggregate one drive's plays and throws into per-route counts and rates."""
    return _rates(_count_routes(plays, throws))


def audit_policy(
    policy: Any,
    scenario_name: str,
    seeds: Iterable[int],
    drive_config: DriveConfig | None = None,
) -> dict[str, Any]:
    """Run instrumented drives for each seed and aggregate the route report."""
    scenario = get_scenario(scenario_name)
    seed_list = list(seeds)
    env = QBDriveEnv(scenario, drive_config)
    drives = [run_instrumented_drive(env, policy, seed) for seed in seed_list]
    env.close()

    totals = _empty_counts()
    plays_total = throws_total = touchdowns = interceptions = turnovers = 0
    points = 0.0
    first_downs = 0.0
    for drive in drives:
        _merge_counts(totals, _count_routes(list(drive.plays), list(drive.records)))
        plays_total += len(drive.plays)
        throws_total += len(drive.records)
        touchdowns += drive.outcome == "touchdown"
        interceptions += drive.outcome == "interception"
        turnovers += drive.outcome == "turnover_on_downs"
        points += drive.points
        first_downs += drive.first_downs

    count = len(drives)
    return {
        "routes": _rates(totals),
        "drives": {
            "count": count,
            "plays": plays_total,
            "throws": throws_total,
            "touchdowns": touchdowns,
            "interceptions": interceptions,
            "turnover_on_downs": turnovers,
            "points_per_drive": points / count if count else 0.0,
            "first_downs_per_drive": first_downs / count if count else 0.0,
        },
    }
