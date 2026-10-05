"""Balance guards for the shotgun depth run fix.

The full 100-drive report lives in ``scripts/qb_run_feasibility.py``; this test
uses a small fixed-seed sample of the run-only probe to catch the regression
that motivated the fix: a quarterback starting on the line of scrimmage can
move the chains and score on every drive just by running forward.
"""

from dataclasses import replace

import numpy as np

from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.routes import RouteName
from qb_sim.rules.scenarios import STAGE_B_CONFIG, Scenario
from qb_sim.training.drive_env import QBDriveEnv

EPISODES = 5
SEED_START = 20_000
RUSHER_STARTS = ((27.0, 29.0),)
RUN_ACTION = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def _run_only(depth: float) -> dict[str, float]:
    config = replace(
        STAGE_B_CONFIG,
        qb_snap_depth_yards=depth,
        rusher_starts=RUSHER_STARTS,
        qb_run_speed_factor=1.0,
    )
    scenario = Scenario(f"guard_depth_{depth:g}", config, (RouteName.POST, RouteName.SLANT))
    env = QBDriveEnv(scenario)
    rushes = 0
    rush_yards = 0.0
    first_down_drives = 0
    touchdowns = 0
    try:
        for seed in range(SEED_START, SEED_START + EPISODES):
            env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(RUN_ACTION)
            rush_plays = [
                play for play in env.play_history if play["outcome"] == PlayOutcome.RUSH.value
            ]
            rushes += len(rush_plays)
            rush_yards += sum(play["yards"] for play in rush_plays)
            first_down_drives += int(env.drive.state.first_downs > 0)
            touchdowns += int(env.drive.state.end_reason == "touchdown")
    finally:
        env.close()
    return {
        "rushes": rushes,
        "yards_per_rush": rush_yards / rushes if rushes else 0.0,
        "first_down_drives": first_down_drives,
        "touchdowns": touchdowns,
    }


def test_shotgun_depth_stops_the_free_running_quarterback() -> None:
    on_the_line = _run_only(0.0)
    shotgun = _run_only(3.0)

    assert on_the_line["touchdowns"] > 0
    assert shotgun["touchdowns"] == 0
    assert shotgun["first_down_drives"] == 0
    assert shotgun["yards_per_rush"] < on_the_line["yards_per_rush"]
