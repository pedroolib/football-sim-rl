"""Live training dashboard with an optional training subprocess.

Example (single command demo):

    python scripts/train_dashboard.py --train --run-name ppo_default_v2

Record the window with OBS or any screen recorder; press Esc to exit.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from qb_sim.rules.scenarios import SCENARIOS
from qb_sim.training.monitoring import run_has_training_artifacts
from qb_sim.ui.dashboard import TrainingDashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="QB Lab live training dashboard")
    parser.add_argument("--run-name", default="ppo_default_v2")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--scenario", default="rushed", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--env",
        choices=("passing", "drive", "role_play", "role_drive"),
        default="passing",
    )
    parser.add_argument("--learner", default=None)
    parser.add_argument(
        "--opponent",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Frozen opponent policy for role modes; repeatable.",
    )
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--train", action="store_true", help="Launch training as a subprocess")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --train to overwrite an existing run directory",
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    default_learners = {
        "passing": "qb",
        "drive": "qb",
        "role_play": "wr1",
        "role_drive": "cb1",
    }
    learner = args.learner or default_learners[args.env]
    opponents: list[tuple[str, str]] = []
    for entry in args.opponent:
        parts = entry.split("=", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            parser.error(f"--opponent expects ID=PATH, got {entry!r}")
        opponents.append((parts[0], parts[1]))
    opponent_policies = None
    if args.env in ("role_play", "role_drive"):
        from qb_sim.training.train import build_opponents

        opponent_policies = build_opponents(args.env, args.scenario, opponents)

    run_dir = Path(args.output_dir) / args.run_name
    if args.train and run_has_training_artifacts(run_dir) and not args.force:
        parser.error(
            f"{run_dir} ya tiene un entrenamiento guardado (config.json).\n"
            "  - Para verlo sin entrenar: quitá --train.\n"
            "  - Para entrenar algo nuevo: usá otro --run-name.\n"
            "  - Para reentrenar sobre este run: agregá --force."
        )
    process = None
    if args.train:
        command = [
            sys.executable,
            "-m",
            "qb_sim.training.train",
            "--timesteps",
            str(args.timesteps),
            "--seed",
            str(args.seed),
            "--run-name",
            args.run_name,
            "--output-dir",
            args.output_dir,
            "--scenario",
            args.scenario,
            "--env",
            args.env,
            "--device",
            args.device,
        ]
        if args.env in ("role_play", "role_drive"):
            command.extend(["--learner", learner])
            for player_id, path in opponents:
                command.extend(["--opponent", f"{player_id}={path}"])
        process = subprocess.Popen(command)
    try:
        TrainingDashboard(
            run_dir,
            scenario_name=args.scenario,
            panels=args.panels,
            env_kind=args.env,
            learner=learner,
            opponent_policies=opponent_policies,
            training=args.train,
        ).run()
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
