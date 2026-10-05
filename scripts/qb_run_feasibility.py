"""Feasibility probe for QB run balance at different shotgun depths.

For each candidate snap depth it runs two seeded ``QBDriveEnv`` probes against
the rule defense: a run-only quarterback that always moves forward, and the
rule quarterback. The first shows how many free yards the run gives up as the
quarterback starts further behind the line; the second checks that passing
stays viable and when the rusher arrives.

Example:

    python scripts/qb_run_feasibility.py --episodes 100
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from qb_sim.engine.entities import PlayOutcome
from qb_sim.engine.geometry import distance
from qb_sim.rules.routes import RouteName
from qb_sim.rules.scenarios import STAGE_B_CONFIG, Scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.evaluation import RuleQuarterbackPolicy

DEFAULT_EPISODES = 100
DEFAULT_SEED_START = 20_000
DEPTHS = (0.0, 3.0, 4.0, 5.0)
RUSHER_STARTS = ((27.0, 29.0),)
RUN_ACTION = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def build_scenario(depth: float) -> Scenario:
    """Stage B roster with a custom shotgun depth and the LOS+7 rusher."""
    config = replace(
        STAGE_B_CONFIG,
        qb_snap_depth_yards=float(depth),
        rusher_starts=RUSHER_STARTS,
    )
    return Scenario(
        f"qb_run_depth_{depth:g}",
        config,
        (RouteName.POST, RouteName.SLANT),
    )


def measure_run_drive(env: QBDriveEnv, seed: int) -> dict:
    """Run one drive with a quarterback that always moves forward."""
    env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(RUN_ACTION)
    plays = env.play_history
    rushes = [play for play in plays if play["outcome"] == PlayOutcome.RUSH.value]
    return {
        "plays": len(plays),
        "rushes": len(rushes),
        "rush_yards": float(sum(play["yards"] for play in rushes)),
        "first_downs": env.drive.state.first_downs,
        "points": env.drive.state.points,
        "touchdown": env.drive.state.end_reason == "touchdown",
    }


def summarize_run(rows: list[dict]) -> dict:
    """Aggregate the run-only probe over drives."""
    total = len(rows)
    rushes = sum(row["rushes"] for row in rows)
    rush_yards = sum(row["rush_yards"] for row in rows)
    first_down_drives = sum(row["first_downs"] > 0 for row in rows)
    touchdowns = sum(row["touchdown"] for row in rows)
    return {
        "episodes": total,
        "rushes": rushes,
        "yards_per_rush": rush_yards / rushes if rushes else 0.0,
        "first_down_drive_pct": 100.0 * first_down_drives / total,
        "touchdown_pct": 100.0 * touchdowns / total,
        "points_per_drive": sum(row["points"] for row in rows) / total,
    }


def measure_rule_drive(env: QBDriveEnv, policy: RuleQuarterbackPolicy, seed: int) -> dict:
    """Run one rule-QB drive and record the rusher arrival tick."""
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    arrival_ticks: int | None = None
    ticks = 0
    while not (terminated or truncated):
        play = env.play
        rusher_id = play.rusher_ids[0] if play.rusher_ids else None
        observation, _, terminated, truncated, _ = env.step(policy(env, observation))
        ticks += 1
        if arrival_ticks is None and rusher_id is not None:
            state = play.state
            if (
                distance(state.player(rusher_id).position, state.player("qb").position)
                <= play.config.tackle_radius
            ):
                arrival_ticks = ticks
    plays = env.play_history
    return {
        "plays": len(plays),
        "sacks": sum(play["outcome"] == PlayOutcome.SACK.value for play in plays),
        "points": env.drive.state.points,
        "touchdown": env.drive.state.end_reason == "touchdown",
        "arrival_ticks": arrival_ticks,
    }


def summarize_rule(rows: list[dict], dt: float) -> dict:
    """Aggregate the rule-QB probe over drives."""
    total = len(rows)
    plays = sum(row["plays"] for row in rows)
    sacks = sum(row["sacks"] for row in rows)
    touchdowns = sum(row["touchdown"] for row in rows)
    arrivals = [row["arrival_ticks"] for row in rows if row["arrival_ticks"] is not None]
    return {
        "episodes": total,
        "touchdown_pct": 100.0 * touchdowns / total,
        "points_per_drive": sum(row["points"] for row in rows) / total,
        "sacks_per_play": sacks / plays if plays else 0.0,
        "rusher_arrival_pct": 100.0 * len(arrivals) / total,
        "mean_rusher_arrival_ticks": float(np.mean(arrivals)) if arrivals else None,
        "mean_rusher_arrival_seconds": float(np.mean(arrivals)) * dt if arrivals else None,
    }


def probe_depth(depth: float, episodes: int, seed_start: int) -> dict:
    """Run both probes for one depth over the same seed range."""
    scenario = build_scenario(depth)
    seeds = range(seed_start, seed_start + episodes)
    env = QBDriveEnv(scenario)
    try:
        run_rows = [measure_run_drive(env, seed) for seed in seeds]
        policy = RuleQuarterbackPolicy(scenario.config)
        rule_rows = [measure_rule_drive(env, policy, seed) for seed in seeds]
        dt = scenario.config.dt
    finally:
        env.close()
    return {
        "depth": depth,
        "run_only": summarize_run(run_rows),
        "rule_qb": summarize_rule(rule_rows, dt),
    }


def _format(value: float | None, width: int = 8, decimals: int = 2) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="QB run feasibility by shotgun depth")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--output", default="runs/reports/qb_run_feasibility.json")
    args = parser.parse_args()

    results = [
        probe_depth(depth, args.episodes, args.seed_start) for depth in DEPTHS
    ]

    print(
        f"QB run feasibility ({args.episodes} drives/depth, seeds {args.seed_start}+, "
        f"rusher {RUSHER_STARTS[0]})"
    )
    print(
        f"{'depth':>6}{'yds/rush':>10}{'1stD%':>8}{'runTD%':>8}"
        f"{'ruleTD%':>9}{'pts/drive':>11}{'sack/play':>11}{'arr(s)':>8}{'arr%':>7}"
    )
    for entry in results:
        run = entry["run_only"]
        rule = entry["rule_qb"]
        print(
            f"{entry['depth']:>6.1f}"
            f"{run['yards_per_rush']:>10.2f}"
            f"{run['first_down_drive_pct']:>7.1f}%"
            f"{run['touchdown_pct']:>7.1f}%"
            f"{rule['touchdown_pct']:>8.1f}%"
            f"{rule['points_per_drive']:>11.2f}"
            f"{rule['sacks_per_play']:>11.3f}"
            f"{_format(rule['mean_rusher_arrival_seconds'])}"
            f"{rule['rusher_arrival_pct']:>6.1f}%"
        )

    report = {
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "rusher_starts": RUSHER_STARTS,
        "depths": DEPTHS,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
