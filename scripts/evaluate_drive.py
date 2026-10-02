"""Compare the rule-based and trained quarterbacks on fixed drive seeds.

Example:

    python scripts/evaluate_drive.py \
        --ppo v11=runs/ppo_drive_qb_v11/final_model.zip \
        --episodes 100 --output runs/drive_evaluation_v11.json
"""

import argparse
import json
from pathlib import Path

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.drive_evaluation import evaluate_drive
from qb_sim.training.evaluation import PPOActionPolicy, RuleQuarterbackPolicy

DEFAULT_EPISODES = 100
DEFAULT_SEED_START = 10_000
COLUMNS = (
    "touchdown_pct",
    "points_per_drive",
    "first_downs_per_drive",
    "plays_per_drive",
    "third_down_conversion_pct",
    "third_and_long_conversion_pct",
    "third_and_long_throw_depth_yards",
    "checkdown_pct",
    "interception_pct",
    "sack_pct",
    "coverage_sack_pct",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate quarterbacks on QBDrive-v0")
    parser.add_argument("--ppo", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--scenario", default="rushed")
    parser.add_argument("--output", default="runs/drive_evaluation.json")
    args = parser.parse_args()

    scenario = get_scenario(args.scenario)
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    policies: list[tuple[str, object]] = [
        ("rules", RuleQuarterbackPolicy(scenario.config))
    ]
    for entry in args.ppo:
        label, path = entry.split("=", 1)
        policies.append((label, PPOActionPolicy(path)))

    results: dict[str, dict] = {}
    for label, policy in policies:
        if isinstance(policy, PPOActionPolicy) and not policy.supports_drive_layout():
            print(f"{label}: omitido (layout incompatible: {policy.observation_size})")
            continue
        results[label] = evaluate_drive(policy, args.scenario, seeds)

    print(f"\n=== {args.scenario} ({args.episodes} drives, semillas {args.seed_start}+) ===")
    header = f"{'policy':<12}" + "".join(f"{column[:11]:>12}" for column in COLUMNS)
    print(header)
    for name, summary in results.items():
        row = f"{name:<12}" + "".join(f"{summary[column]:>12.2f}" for column in COLUMNS)
        print(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
