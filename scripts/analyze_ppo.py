"""Behavior analysis for a trained quarterback policy.

Example:

    python scripts/analyze_ppo.py --model runs/ppo_default/final_model.zip \
        --episodes 100 --output runs/analysis.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.env import QBPassingEnv
from qb_sim.training.evaluation import PPOActionPolicy, run_episode, summarize_episodes

DEFAULT_EPISODES = 100
DEFAULT_SEED_START = 10_000
MOVE_SPEED_THRESHOLD = 0.5
HISTOGRAM_BINS = 6


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a trained QB policy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--output", default="runs/analysis.json")
    args = parser.parse_args()

    scenario = get_scenario(args.scenario)
    env = QBPassingEnv(scenario)
    policy = PPOActionPolicy(args.model)
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    results = [run_episode(env, policy, seed) for seed in seeds]
    summary = summarize_episodes(results)

    throws = [result for result in results if result.throw_time is not None]
    throw_times = [result.throw_time for result in throws]
    moving_throws = [
        result
        for result in throws
        if result.qb_speed_at_throw is not None and result.qb_speed_at_throw > MOVE_SPEED_THRESHOLD
    ]
    counts, edges = np.histogram(
        throw_times,
        bins=HISTOGRAM_BINS,
        range=(0.0, scenario.config.max_play_duration),
    )

    analysis = {
        **summary,
        "moving_throw_pct": 100.0 * len(moving_throws) / len(throws) if throws else 0.0,
        "throw_time_histogram": {
            f"{edges[index]:0.2f}-{edges[index + 1]:0.2f}s": int(count)
            for index, count in enumerate(counts)
        },
    }

    print(f"=== {args.scenario} ({args.episodes} episodios) ===")
    print(f"lanzamientos: {summary['throw_frequency']:.1f}% de episodios")
    print(f"tiempo medio de lanzamiento: {summary['average_throw_time']:.2f}s")
    print(f"presión media al lanzar: {summary['average_pressure_at_throw']:.2f}")
    print(f"velocidad media del QB al lanzar: {summary['average_qb_speed_at_throw']:.2f} yd/s")
    print(f"lanzamientos en movimiento: {analysis['moving_throw_pct']:.1f}%")
    print(f"movimiento medio por jugada: {summary['move_frequency']:.1f}% de ticks")
    print("histograma de tiempo de lanzamiento:")
    for bucket, count in analysis["throw_time_histogram"].items():
        print(f"  {bucket:>10}  {count}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
