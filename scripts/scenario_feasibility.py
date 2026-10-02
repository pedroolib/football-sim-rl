"""Rule-QB feasibility report for the M7 speed scenarios and randomization.

Run from the repository root:

    python scripts/scenario_feasibility.py
"""

import numpy as np

from qb_sim.engine.geometry import distance
from qb_sim.rules.scenarios import RUSHED_SCENARIO, SCENARIOS
from qb_sim.training.env import QBPassingEnv
from qb_sim.training.evaluation import RuleQuarterbackPolicy
from qb_sim.training.randomization import RandomizedEnv, SpeedRanges

EPISODES = 100
SEED_START = 10_000
RANDOM_CONFIGS = 30
RANDOM_EPISODES = 10


def measure_episode(env, seed: int) -> tuple[str, float | None]:
    """Run one rule episode; return the outcome and the rusher arrival time.

    Arrival is the first time the rusher comes within tackle radius of the
    quarterback, regardless of the ball state; sacks are only counted when the
    engine reports ``PlayOutcome.SACK``.
    """
    observation, _ = env.reset(seed=seed)
    policy = RuleQuarterbackPolicy(env.play.config)
    arrival: float | None = None
    outcome = "in_progress"
    while True:
        state = env.play.state
        if arrival is None:
            for rusher_id in env.play.rusher_ids:
                rusher = state.player(rusher_id)
                if (
                    distance(rusher.position, state.player("qb").position)
                    <= env.play.config.tackle_radius
                ):
                    arrival = state.elapsed_seconds
        action = policy(env, observation)
        observation, _, terminated, truncated, info = env.step(action)
        outcome = info["outcome"]
        if terminated or truncated:
            return outcome, arrival


def summarize(rows: list[tuple[str, float | None]]) -> dict[str, float | None]:
    total = len(rows)
    completed = sum(1 for outcome, _ in rows if outcome == "completed")
    sacks = sum(1 for outcome, _ in rows if outcome == "sack")
    interceptions = sum(1 for outcome, _ in rows if outcome == "interception")
    arrivals = [arrival for _, arrival in rows if arrival is not None]
    return {
        "episodes": total,
        "completion_pct": 100.0 * completed / total,
        "sack_pct": 100.0 * sacks / total,
        "interception_pct": 100.0 * interceptions / total,
        "mean_arrival": float(np.mean(arrivals)) if arrivals else None,
        "min_arrival": float(np.min(arrivals)) if arrivals else None,
    }


def main() -> None:
    print("Escenarios con rusher (QB de reglas, 100 episodios, semillas 10000+)")
    print(
        f"{'scenario':<16}{'completos':>10}{'sacks':>8}{'intercep':>10}{'llegada_med':>13}{'llegada_min':>13}"
    )
    for name, scenario in SCENARIOS.items():
        if not scenario.config.rusher_starts:
            continue
        env = QBPassingEnv(scenario)
        rows = [measure_episode(env, seed) for seed in range(SEED_START, SEED_START + EPISODES)]
        summary = summarize(rows)
        arrival_mean = f"{summary['mean_arrival']:.2f}" if summary["mean_arrival"] else "-"
        arrival_min = f"{summary['min_arrival']:.2f}" if summary["min_arrival"] else "-"
        print(
            f"{name:<16}{summary['completion_pct']:>9.1f}%{summary['sack_pct']:>7.1f}%"
            f"{summary['interception_pct']:>9.1f}%{arrival_mean:>13}{arrival_min:>13}"
        )
        env.close()

    print()
    print(f"Randomización (30 configuraciones × 10 episodios, {SpeedRanges()})")
    ranges = SpeedRanges()
    completions: list[float] = []
    sacks: list[float] = []
    arrivals: list[float] = []
    for config_index in range(RANDOM_CONFIGS):
        env = RandomizedEnv(RUSHED_SCENARIO, ranges)
        rows = [
            measure_episode(env, config_index * 1_000 + SEED_START + episode)
            for episode in range(RANDOM_EPISODES)
        ]
        summary = summarize(rows)
        completions.append(summary["completion_pct"] or 0.0)
        sacks.append(summary["sack_pct"] or 0.0)
        if summary["min_arrival"] is not None:
            arrivals.append(summary["min_arrival"])
        env.close()

    print(
        f"completos: min {np.min(completions):.1f}%  media {np.mean(completions):.1f}%  "
        f"max {np.max(completions):.1f}%"
    )
    print(f"sacks: max {np.max(sacks):.1f}%  media {np.mean(sacks):.1f}%")
    if arrivals:
        print(f"llegada del rusher: min {np.min(arrivals):.2f}s  media {np.mean(arrivals):.2f}s")


if __name__ == "__main__":
    main()
