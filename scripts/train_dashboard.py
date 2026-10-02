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
from qb_sim.ui.dashboard import TrainingDashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="QB Lab live training dashboard")
    parser.add_argument("--run-name", default="ppo_default_v2")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--scenario", default="rushed", choices=sorted(SCENARIOS))
    parser.add_argument("--env", choices=("passing", "drive"), default="passing")
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--train", action="store_true", help="Launch training as a subprocess")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = Path(args.output_dir) / args.run_name
    process = None
    if args.train:
        process = subprocess.Popen(
            [
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
        )
    try:
        TrainingDashboard(
            run_dir,
            scenario_name=args.scenario,
            panels=args.panels,
            env_kind=args.env,
        ).run()
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
