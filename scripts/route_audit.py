"""Data-driven route audit for drive quarterbacks (Fase 0, E0.4).

Example:

    python scripts/route_audit.py \
        --policy rules=rules \
        --policy qb=runs/qb/drive/teacher_vs_wr_pair/final_model.zip \
        --scenario stage_b --episodes 100 --seed-start 20000 \
        --output runs/reports/route_audit.json
"""

import argparse
import json
from pathlib import Path

from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER
from qb_sim.rules.scenarios import Scenario, get_scenario
from qb_sim.training.drive_observations import DRIVE_OBSERVATION_SIZE
from qb_sim.training.evaluation import PPOActionPolicy, RuleQuarterbackPolicy
from qb_sim.training.route_audit import audit_policy

DEFAULT_EPISODES = 100
DEFAULT_SEED_START = 20_000
COLUMNS = ("sampled", "target_pct", "completion_pct", "yards_per_play", "int_pct", "td_pct")
HEADINGS = ("sampled", "target%", "comp%", "yds/play", "INT%", "TD%")


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


def _print_table(label: str, report: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"{'route':<8}" + "".join(f"{heading:>11}" for heading in HEADINGS))
    for route in ROUTE_OBSERVATION_ORDER:
        stats = report["routes"][route.value]
        row = f"{route.value:<8}"
        for key in COLUMNS:
            value = stats[key]
            row += f"{value:>11.0f}" if key == "sampled" else f"{value:>11.2f}"
        print(row)
    drives = report["drives"]
    print(
        f"{drives['count']} drives, {drives['plays']} jugadas, "
        f"{drives['touchdowns']} TD, {drives['interceptions']} INT, "
        f"{drives['points_per_drive']:.2f} pts/drive"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoría de rutas por situación")
    parser.add_argument("--policy", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--scenario", default="stage_b")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--output", default="runs/reports/route_audit.json")
    args = parser.parse_args()
    if not args.policy:
        parser.error("at least one --policy LABEL=PATH is required")

    scenario = get_scenario(args.scenario)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    policies: list[tuple[str, object]] = []
    for entry in args.policy:
        label, separator, path = entry.partition("=")
        if not separator or not path:
            parser.error(f"--policy requires LABEL=PATH, got {entry!r}")
        policies.append((label, _build_policy(label, path, scenario)))

    print(f"\n=== {args.scenario} ({args.episodes} drives, semillas {args.seed_start}+) ===")
    reports: dict[str, dict] = {}
    for label, policy in policies:
        reports[label] = audit_policy(policy, args.scenario, seeds)
        _print_table(label, reports[label])

    payload = {
        "scenario": args.scenario,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "policies": reports,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
