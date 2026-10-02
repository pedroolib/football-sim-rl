"""Calibrate carrier fatigue: TD rate and YAC by receiver-corner speed gap.

Example:

    python scripts/carrier_feasibility.py --fatigue 0.12 --drives 100
"""

import argparse
from dataclasses import replace

import numpy as np

from qb_sim.agents.loop import build_rule_agents
from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.drive import Drive, DriveConfig
from qb_sim.rules.scenarios import get_scenario

DEFAULT_SEEDS = 100
GAPS = ((7.0, 7.0), (7.0, 6.7), (7.5, 7.0), (7.5, 6.8), (8.0, 7.0))


def run_drive(config, seed, drive_config):
    drive = Drive(config, drive_config, seed=seed)
    touchdowns = 0
    catches = 0
    yac = []
    while not drive.state.terminated:
        play = drive.start_play()
        agents = build_rule_agents(play, drive.routes, config, include_quarterback=True)
        while play.state.outcome is PlayOutcome.IN_PROGRESS:
            actions = {}
            for player_id, agent in agents.items():
                action = agent.decide(play.state)
                if action is not None:
                    actions[player_id] = action
            play.step(actions)
        carrier = play.state.carrier_id
        if carrier is not None:
            catches += 1
            yac.append(float(play.state.player(carrier).position[0] - drive.state.los))
        result = drive.advance(play)
        touchdowns += int(result.touchdown)
    return touchdowns, catches, yac


def main() -> None:
    parser = argparse.ArgumentParser(description="Carrier fatigue feasibility")
    parser.add_argument("--fatigue", type=float, default=0.12)
    parser.add_argument("--floor", type=float, default=0.55)
    parser.add_argument("--drives", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed-start", type=int, default=10_000)
    args = parser.parse_args()

    drive_config = DriveConfig(los_range=(40.0, 40.0))
    print(f"fatigue={args.fatigue}/s floor={args.floor} drives/gap={args.drives}")
    print(f"{'recv':>5} {'cb':>5} {'td%':>7} {'catches':>8} {'yac':>7}")
    for receiver_speed, defender_speed in GAPS:
        config = replace(
            get_scenario("stage_b").config,
            receiver_speed=receiver_speed,
            defender_speed=defender_speed,
            carrier_fatigue_per_second=args.fatigue,
            carrier_min_speed_factor=args.floor,
        )
        touchdowns = 0
        catches = 0
        yac: list[float] = []
        for seed in range(args.seed_start, args.seed_start + args.drives):
            td, caught, yards = run_drive(config, seed, drive_config)
            touchdowns += td
            catches += caught
            yac.extend(yards)
        mean_yac = float(np.mean(yac)) if yac else 0.0
        print(
            f"{receiver_speed:>5.1f} {defender_speed:>5.1f} "
            f"{100.0 * touchdowns / args.drives:>6.1f}% {catches:>8} {mean_yac:>7.2f}"
        )


if __name__ == "__main__":
    main()
