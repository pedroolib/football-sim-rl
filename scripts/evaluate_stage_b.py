"""Evaluate Stage B roles and the frozen composition on fixed seeds.

Examples:

    python scripts/evaluate_stage_b.py wr --episodes 100 \
        --ppo wr1=runs/ppo_wr_v1/final_model.zip \
        --opponent qb=runs/ppo_drive_qb_v11/final_model.zip
    python scripts/evaluate_stage_b.py composition --episodes 100 \
        --ppo qb=runs/ppo_drive_qb_v12/final_model.zip \
        --opponent wr1=runs/ppo_wr_v1/final_model.zip \
        --opponent cb1=runs/ppo_cb_v1/final_model.zip

The ``cb`` and ``ru`` modes require ``--opponent qb=PATH``: without a
quarterback opponent the QB stays passive and the metrics are meaningless.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.evaluation import PPOActionPolicy, RuleQuarterbackPolicy
from qb_sim.training.role_evaluation import (
    evaluate_composition,
    evaluate_role_drive,
    evaluate_role_play,
)
from qb_sim.training.train import build_opponents

LEARNERS = {"wr": "wr1", "cb": "cb1", "ru": "ru1"}
ENV_KINDS = {"composition": "drive", "wr": "role_play", "cb": "role_drive", "ru": "role_drive"}
ALLOWED_OPPONENTS = {
    "composition": ("qb", "wr1", "wr2", "cb1", "cb2", "ru1"),
    "wr": ("qb", "wr1", "wr2"),
    "cb": ("qb", "wr1", "wr2", "cb1", "cb2", "ru1"),
    "ru": ("qb", "wr1", "wr2", "cb1", "cb2", "ru1"),
}


def _pairs(entries: list[str], parser: argparse.ArgumentParser) -> tuple[tuple[str, str], ...]:
    pairs = []
    for entry in entries:
        if "=" not in entry:
            parser.error(f"expected ID=PATH, got {entry!r}")
        pairs.append(tuple(entry.split("=", 1)))
    return tuple(pairs)


def _rule_role_action(env, observation) -> np.ndarray:
    action = env.rule_agent(env.learner_id).decide(env.play.state)
    if action is None:
        return np.zeros(2, dtype=np.float32)
    return np.array([action.direction[0], action.direction[1]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B evaluation")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("composition", "wr", "cb", "ru"):
        sub = subparsers.add_parser(mode)
        sub.add_argument("--ppo", action="append", default=[], metavar="LABEL=PATH")
        sub.add_argument("--opponent", action="append", default=[], metavar="ID=PATH")
        sub.add_argument("--episodes", type=int, default=100)
        sub.add_argument("--seed-start", type=int, default=10_000)
        sub.add_argument("--scenario", default="stage_b")
        sub.add_argument("--output", default=None)
    args = parser.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.episodes)
    scenario = get_scenario(args.scenario)
    env_kind = ENV_KINDS[args.mode]
    ppo_pairs = _pairs(args.ppo, parser)
    opponent_pairs = _pairs(args.opponent, parser)
    allowed = ALLOWED_OPPONENTS[args.mode]
    for player_id, _ in opponent_pairs:
        if player_id not in allowed:
            parser.error(
                f"{args.mode} mode cannot use opponent {player_id!r}; "
                f"allowed ids: {', '.join(allowed)}"
            )
    if args.mode in {"cb", "ru"} and "qb" not in dict(opponent_pairs):
        parser.error(f"{args.mode} mode requires a quarterback opponent: --opponent qb=PATH")
    opponents = build_opponents(env_kind, args.scenario, opponent_pairs)
    results: dict[str, dict] = {}

    if args.mode == "composition":
        results["rules"] = evaluate_composition(
            RuleQuarterbackPolicy(scenario.config), opponents, args.scenario, seeds
        )
        for label, path in ppo_pairs:
            results[label] = evaluate_composition(
                PPOActionPolicy(path), opponents, args.scenario, seeds
            )
    else:
        learner = LEARNERS[args.mode]
        env_kwargs = {"learner_id": learner, "opponents": opponents}
        evaluate = evaluate_role_play if args.mode == "wr" else evaluate_role_drive
        results["rules"] = evaluate(_rule_role_action, args.scenario, seeds, **env_kwargs)
        for label, path in ppo_pairs:
            results[label] = evaluate(PPOActionPolicy(path), args.scenario, seeds, **env_kwargs)

    for name, summary in results.items():
        print(f"\n{args.mode} — {name}")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    output = Path(args.output or f"runs/stage_b_evaluation_{args.mode}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": args.mode,
        "scenario": args.scenario,
        "seed_start": args.seed_start,
        "episodes": args.episodes,
        "ppo": {label: path for label, path in ppo_pairs},
        "opponents": {player_id: path for player_id, path in opponent_pairs},
        "output": str(output),
    }
    payload = {"metadata": metadata, "results": results}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
