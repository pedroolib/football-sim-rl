"""Reproducible baseline metrics for rule-based teams.

Run from the repository root:

    python scripts/rule_baseline_report.py
"""

from qb_sim.agents.loop import run_series
from qb_sim.metrics import summarize
from qb_sim.rules.scenarios import SCENARIOS

PLAYS_PER_SCENARIO = 200
METRICS = (
    "completion_pct",
    "interception_pct",
    "incompletion_pct",
    "out_of_bounds_pct",
    "time_limit_pct",
    "yards_per_attempt",
    "yards_per_completion",
    "average_time_to_throw",
    "average_duration",
)


def main() -> None:
    for scenario in SCENARIOS.values():
        results = run_series(scenario, count=PLAYS_PER_SCENARIO)
        summary = summarize(results, line_of_scrimmage=scenario.config.quarterback_start[0])
        print(f"{scenario.name} ({PLAYS_PER_SCENARIO} jugadas, semillas fijas)")
        for name in METRICS:
            print(f"  {name:<22} {summary[name]:8.2f}")
        print()


if __name__ == "__main__":
    main()
