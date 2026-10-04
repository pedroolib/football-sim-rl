"""Evaluate the Stage B2 adaptation matrix over capability cells.

Each cell sets the receiver speed, the defender gap, the throw range, and the
throw sigma, then evaluates the rule baseline and the trained composition
(QB + frozen generalized roles) on the same seeds.

Example:

    python scripts/evaluate_adaptation.py \
        --qb runs/ppo_qb_g1/final_model.zip \
        --wr runs/ppo_wr_g1/final_model.zip \
        --cb runs/ppo_cb_g1/final_model.zip \
        --ru runs/ppo_ru_g1/final_model.zip \
        --gaps 0.0,0.5,1.0 --ranges 35,45,55 --sigmas 1.0 \
        --episodes 100 --seed-start 10000 --output runs/stage_b2_matrix.json
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from qb_sim.config import SimulationConfig
from qb_sim.rules.routes import RouteName
from qb_sim.rules.scenarios import STAGE_B_CONFIG, Scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.evaluation import PPOActionPolicy, RuleQuarterbackPolicy
from qb_sim.training.policy import load_defender_policy, load_receiver_policy
from qb_sim.training.role_evaluation import run_composition_episode, summarize_composition

RULE_LABEL = "rules"
TRAINED_LABEL = "policies"


def _floats(raw: str) -> list[float]:
    return [float(token) for token in raw.split(",") if token]


def _ints(raw: str) -> list[int]:
    return [int(token) for token in raw.split(",") if token]


def cell_key(gap: float, throw_range: int, sigma: float) -> str:
    """Stable cell identifier, e.g. ``gap0.5_range45_sigma1.0``."""
    return f"gap{gap}_range{throw_range}_sigma{sigma}"


def cell_config(gap: float, throw_range: int, sigma: float) -> SimulationConfig:
    """Capability configuration for one matrix cell."""
    return replace(
        STAGE_B_CONFIG,
        receiver_speed=7.0,
        defender_speed=7.0 - gap,
        quarterback_speed=6.5,
        throw_max_range=float(throw_range),
        throw_base_sigma_degrees=sigma,
    )


def cell_scenario(gap: float, throw_range: int, sigma: float) -> Scenario:
    """Named scenario wrapping the cell configuration."""
    return Scenario(
        "adaptation",
        cell_config(gap, throw_range, sigma),
        (RouteName.POST, RouteName.SLANT),
    )


def build_opponents(args: argparse.Namespace) -> dict[str, Any]:
    """Load the frozen generalized roles with the base stage_b configuration.

    WR1/WR2 share the receiver model, CB1/CB2 share the defender model, and RU1
    uses the rusher model. The environment pushes each cell's active config into
    these adapters on reset through ``set_config``.
    """
    return {
        "wr1": load_receiver_policy(args.wr, "stage_b", "wr1"),
        "wr2": load_receiver_policy(args.wr, "stage_b", "wr2"),
        "cb1": load_defender_policy(args.cb, "stage_b", "cb1"),
        "cb2": load_defender_policy(args.cb, "stage_b", "cb2"),
        "ru1": load_defender_policy(args.ru, "stage_b", "ru1"),
    }


def evaluate_cell(
    scenario: Scenario,
    opponents: dict[str, Any],
    qb_policy,
    seeds: range,
) -> dict:
    """Run one cell with a QB policy and summarize the composition."""
    env = QBDriveEnv(scenario, opponents=opponents)
    results = [run_composition_episode(env, qb_policy, seed) for seed in seeds]
    env.close()
    return summarize_composition(results)


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def print_table(cells: dict[str, dict]) -> None:
    header = (
        f"{'cell':<26}{'policy':<10}{'TD%':>7}{'comp%':>7}"
        f"{'throw_yd':>10}{'scram%':>8}{'t_throw':>9}{'INT%':>7}"
    )
    print(header)
    for key, cell in cells.items():
        for label in (RULE_LABEL, TRAINED_LABEL):
            summary = cell[label]
            print(
                f"{key:<26}{label:<10}"
                f"{_number(summary['touchdown_pct']):>7}"
                f"{_number(summary['completion_pct']):>7}"
                f"{_number(summary['mean_throw_distance']):>10}"
                f"{_number(summary['mean_scramble_pct']):>8}"
                f"{_number(summary['mean_time_to_throw']):>9}"
                f"{_number(summary['interception_pct']):>7}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B2 adaptation matrix")
    parser.add_argument("--qb", required=True, help="generalized QB model path")
    parser.add_argument("--wr", required=True, help="generalized receiver model path")
    parser.add_argument("--cb", required=True, help="generalized defender model path")
    parser.add_argument("--ru", required=True, help="generalized rusher model path")
    parser.add_argument("--gaps", default="0.0,0.5,1.0", help="comma-separated speed gaps")
    parser.add_argument("--ranges", default="35,45,55", help="comma-separated throw ranges")
    parser.add_argument("--sigmas", default="1.0", help="comma-separated sigma degrees")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--output", default="runs/stage_b2_matrix.json")
    args = parser.parse_args()

    gaps = _floats(args.gaps)
    throw_ranges = _ints(args.ranges)
    sigmas = _floats(args.sigmas)
    seeds = range(args.seed_start, args.seed_start + args.episodes)

    trained_opponents = build_opponents(args)
    qb_policy = PPOActionPolicy(args.qb)

    cells: dict[str, dict] = {}
    for gap in gaps:
        for throw_range in throw_ranges:
            for sigma in sigmas:
                scenario = cell_scenario(gap, throw_range, sigma)
                key = cell_key(gap, throw_range, sigma)
                cells[key] = {
                    "config": {
                        "receiver_speed": scenario.config.receiver_speed,
                        "defender_speed": scenario.config.defender_speed,
                        "quarterback_speed": scenario.config.quarterback_speed,
                        "throw_max_range": scenario.config.throw_max_range,
                        "throw_base_sigma_degrees": scenario.config.throw_base_sigma_degrees,
                    },
                    RULE_LABEL: evaluate_cell(
                        scenario, {}, RuleQuarterbackPolicy(scenario.config), seeds
                    ),
                    TRAINED_LABEL: evaluate_cell(
                        scenario, trained_opponents, qb_policy, seeds
                    ),
                }

    print_table(cells)

    report = {
        "metadata": {
            "gaps": gaps,
            "ranges": throw_ranges,
            "sigmas": sigmas,
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "policies": {
                "qb": args.qb,
                "wr": args.wr,
                "cb": args.cb,
                "ru": args.ru,
            },
        },
        "cells": cells,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
