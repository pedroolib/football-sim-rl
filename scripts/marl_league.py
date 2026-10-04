"""Run the M7-MARL league: train roles in rounds against frozen snapshots.

Example:

    python scripts/marl_league.py \
        --rounds 4 --steps 150000 --n-envs 8 --n-steps 128 \
        --eval-episodes 100 --output runs/league --base runs/league/base.json

Each round trains one role (rotation) against one sampled snapshot per opponent
player id, evaluates the learned composition and the rule baseline on the same
fixed seeds, appends a league table row, and saves the new snapshot. The state
is written to ``<output>/league_state.json`` and can be resumed with ``--resume``.

``--base`` and any resumed state must point at existing model files: the league
evaluator loads every role with ``PPO.load``, so the ``rules`` sentinel is not
supported here (it is rejected before training starts).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from qb_sim.training.league import (
    ROLE_OPPONENTS,
    LeagueConfig,
    SnapshotPool,
    evaluate_league,
    league_row,
    opponent_spec,
    train_round,
)

DEFAULT_BASE: dict[str, str] = {
    "qb": "runs/qb/drive/adaptation_g1/final_model.zip",
    "wr": "runs/wr/adaptation_g1/final_model.zip",
    "cb": "runs/cb/adaptation_g1/final_model.zip",
    "ru": "runs/ru/adaptation_g1/final_model.zip",
}

ROLE_BY_PLAYER: dict[str, str] = {
    "qb": "qb",
    "wr1": "wr",
    "wr2": "wr",
    "cb1": "cb",
    "cb2": "cb",
    "ru1": "ru",
}


def build_pools(state: dict, root: Path, pool_size: int) -> dict[str, SnapshotPool]:
    """Create one snapshot pool per player id, seeded with the current model."""
    pools: dict[str, SnapshotPool] = {}
    for player_id, role in ROLE_BY_PLAYER.items():
        pool = SnapshotPool(root / player_id, pool_size=pool_size)
        pool.add(state["models"][role])
        pools[player_id] = pool
    return pools


def _load_state(path: Path, base_models: dict[str, str], table_path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"round": 0, "models": dict(base_models), "table": str(table_path), "stalled": 0}


def _load_rows(table_path: Path) -> list[dict]:
    if not table_path.is_file():
        return []
    return json.loads(table_path.read_text(encoding="utf-8"))


def _flatten(row: dict) -> dict:
    flat = {
        "round": row["round"],
        "role": row["role"],
        "opponents": ";".join(
            f"{player_id}={path}" for player_id, path in row["opponents"].items()
        ),
        "improved": row["improved"],
    }
    for prefix in ("learned", "rules"):
        flat.update({f"{prefix}_{key}": value for key, value in row[prefix].items()})
    return flat


def write_table(table_path: Path, rows: list[dict]) -> None:
    """Persist the league table as JSON and CSV next to ``table_path``."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    flat_rows = [_flatten(row) for row in rows]
    headers = list(flat_rows[0]) if flat_rows else ["round", "role", "opponents", "improved"]
    with table_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(flat_rows)


def _number(value: object) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _print_row(row: dict) -> None:
    learned = row["learned"]
    rules = row["rules"]
    flag = "mejora" if row["improved"] else "sin mejora"
    print(
        f"r{row['round']} {row['role']}: "
        f"TD {_number(learned.get('touchdown_pct'))}% vs {_number(rules.get('touchdown_pct'))}% | "
        f"pts/drive {_number(learned.get('points_per_drive'))} vs "
        f"{_number(rules.get('points_per_drive'))} | {flag}"
    )


def main(argv: list[str] | None = None) -> int:
    defaults = LeagueConfig()
    parser = argparse.ArgumentParser(description="M7-MARL league over frozen snapshots")
    parser.add_argument("--rounds", type=int, default=defaults.rounds)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--rotation", default=",".join(defaults.rotation))
    parser.add_argument("--scenario", default=defaults.scenario)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--n-envs", type=int, default=defaults.n_envs)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--seed-start", type=int, default=defaults.eval_seed_start)
    parser.add_argument("--output", default=defaults.output)
    parser.add_argument(
        "--base",
        default=None,
        help="JSON mapping role -> existing model path ('rules' is not supported)",
    )
    parser.add_argument("--resume", default=None, help="league state JSON to continue from")
    args = parser.parse_args(argv)

    rotation = tuple(part.strip() for part in args.rotation.split(",") if part.strip())
    if not rotation:
        parser.error("--rotation must contain at least one role")
    unknown_roles = [role for role in rotation if role not in ROLE_OPPONENTS]
    if unknown_roles:
        parser.error(f"unknown rotation roles: {', '.join(unknown_roles)}")
    config = LeagueConfig(
        rounds=args.rounds,
        steps=args.steps,
        rotation=rotation,
        scenario=args.scenario,
        seed=args.seed,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        eval_episodes=args.eval_episodes,
        eval_seed_start=args.seed_start,
        output=args.output,
    )

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.resume) if args.resume else output / "league_state.json"
    base_models = dict(DEFAULT_BASE)
    if args.base is not None:
        base_models.update(json.loads(Path(args.base).read_text(encoding="utf-8")))
    default_table = output / "league_table.json"
    state = _load_state(state_path, base_models, default_table)
    for role in ("qb", "wr", "cb", "ru"):
        model = state["models"].get(role)
        if not isinstance(model, str) or not Path(model).is_file():
            parser.error(f"invalid model path for {role}: {model!r}")
    table_path = Path(state.get("table", default_table))
    state["table"] = str(table_path)
    rows = _load_rows(table_path)
    previous_points = float(rows[-1]["learned"]["points_per_drive"]) if rows else None
    stalled = int(state.get("stalled", 0))
    round_index = int(state.get("round", 0))

    while round_index < config.rounds:
        role = rotation[round_index % len(rotation)]
        rng = np.random.default_rng(config.seed + round_index)
        pools = build_pools(state, output / "pools", config.pool_size)
        opponents = opponent_spec(role, pools, rng)
        final_path = train_round(role, config, opponents, round_index)
        state["models"][role] = str(final_path)
        report = evaluate_league(state["models"], config)
        row = league_row(
            round_index,
            role,
            opponents,
            report["learned"],
            report["rules"],
            previous_points,
        )
        rows.append(row)
        previous_points = report["learned"]["points_per_drive"]
        state["round"] = round_index + 1
        stalled = 0 if row["improved"] else stalled + 1
        state["stalled"] = stalled
        write_table(table_path, rows)
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        _print_row(row)

        round_index += 1
        if not row["improved"] and stalled >= config.patience:
            print(f"sin mejora tras {config.patience} rondas; deteniendo la liga")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
