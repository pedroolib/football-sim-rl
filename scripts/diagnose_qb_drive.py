"""Throw-level diagnosis of drive quarterbacks (Fase 0).

Example:

    python scripts/diagnose_qb_drive.py \
        --policy rules=rules \
        --policy teacher=runs/qb/drive/rule_teacher_v1/final_model.zip \
        --episodes 60 --seed-start 20000 \
        --output runs/reports/qb_diagnosis.json
"""

import argparse
import json
from pathlib import Path

from qb_sim.rules.scenarios import Scenario, get_scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.drive_observations import DRIVE_OBSERVATION_SIZE
from qb_sim.training.evaluation import PPOActionPolicy, RuleQuarterbackPolicy
from qb_sim.training.qb_diagnostics import (
    run_instrumented_drive,
    summarize_diagnostics,
)

DEFAULT_EPISODES = 60
DEFAULT_SEED_START = 20_000
COLUMNS = (
    ("touchdown_pct", "TD%"),
    ("points_per_drive", "pts/drive"),
    ("throw_rate_pct", "throw%"),
    ("mean_throw_time", "t_throw"),
    ("moving_throw_pct", "moving%"),
    ("mean_throw_distance", "depth"),
    ("mean_best_advantage", "best_adv"),
    ("mean_target_advantage", "tgt_adv"),
    ("pct_target_max_advantage", "max_adv%"),
    ("pct_deep_open_throws", "deep%"),
)


def _build_policy(label: str, path: str, scenario: Scenario) -> object:
    if label == "rules":
        return RuleQuarterbackPolicy(scenario.config)
    policy = PPOActionPolicy(path, device="cpu")
    if not policy.supports_drive_layout():
        raise SystemExit(
            f"policy {label!r}: {path} tiene {policy.observation_size} valores de "
            f"observación y no es compatible con el layout de drives "
            f"({DRIVE_OBSERVATION_SIZE})."
        )
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico por lanzamiento del QB de drives")
    parser.add_argument("--policy", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--scenario", default="stage_b")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--output", default="runs/reports/qb_diagnosis.json")
    args = parser.parse_args()
    if not args.policy:
        parser.error("at least one --policy LABEL=PATH is required")

    scenario = get_scenario(args.scenario)
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    policies: list[tuple[str, object]] = []
    for entry in args.policy:
        label, separator, path = entry.partition("=")
        if not separator or not path:
            parser.error(f"--policy requires LABEL=PATH, got {entry!r}")
        policies.append((label, _build_policy(label, path, scenario)))

    summaries: dict[str, dict] = {}
    for label, policy in policies:
        env = QBDriveEnv(get_scenario(args.scenario))
        drives = [run_instrumented_drive(env, policy, seed) for seed in seeds]
        env.close()
        summaries[label] = summarize_diagnostics(drives)

    print(f"\n=== {args.scenario} ({args.episodes} drives, semillas {args.seed_start}+) ===")
    header = f"{'policy':<10}" + "".join(f"{heading:>12}" for _, heading in COLUMNS)
    print(header)
    for label, summary in summaries.items():
        row = f"{label:<10}"
        for key, _ in COLUMNS:
            value = summary[key]
            row += f"{value:>12.2f}" if value is not None else f"{'n/a':>12}"
        print(row)

    payload = {
        "scenario": args.scenario,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "policies": summaries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
