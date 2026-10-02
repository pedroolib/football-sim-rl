"""Compare random, rule-based, and trained PPO policies on fixed seeds.

Example:

    python scripts/evaluate_ppo.py \
        --ppo rules=runs/ppo_default/final_model.zip \
        --episodes 100 --output runs/evaluation.json
"""

import argparse
import json
from pathlib import Path

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.evaluation import (
    PPOActionPolicy,
    RandomPolicy,
    RuleQuarterbackPolicy,
    evaluate,
)

DEFAULT_EPISODES = 100
DEFAULT_SEED_START = 10_000
COLUMNS = (
    "completed_pct",
    "interception_pct",
    "incomplete_pct",
    "yards_per_attempt",
    "mean_reward",
    "average_duration",
    "throw_frequency",
    "move_frequency",
    "average_throw_time",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policies on QBPassing-v0")
    parser.add_argument(
        "--ppo",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Trained model to evaluate; repeatable.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--scenarios", default="default,pressured,deep")
    parser.add_argument("--output", default="runs/evaluation.json")
    args = parser.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.episodes)
    ppo_policies = []
    for entry in args.ppo:
        label, path = entry.split("=", 1)
        ppo_policies.append((label, PPOActionPolicy(path)))

    results: dict[str, dict[str, dict[str, float]]] = {}
    for scenario_name in args.scenarios.split(","):
        scenario = get_scenario(scenario_name)
        scenario_results: dict[str, dict[str, float]] = {
            "random": evaluate(RandomPolicy(seed=args.seed_start), scenario_name, seeds),
            "rules": evaluate(RuleQuarterbackPolicy(scenario.config), scenario_name, seeds),
        }
        for label, policy in ppo_policies:
            scenario_results[label] = evaluate(policy, scenario_name, seeds)
        results[scenario_name] = scenario_results

        print(f"\n=== {scenario_name} ({args.episodes} episodios, semillas {args.seed_start}+) ===")
        header = f"{'policy':<12}" + "".join(f"{column[:11]:>12}" for column in COLUMNS)
        print(header)
        for policy_name, summary in scenario_results.items():
            row = f"{policy_name:<12}" + "".join(f"{summary[column]:>12.2f}" for column in COLUMNS)
            print(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
