"""Configurable reward function for the quarterback environment."""

from dataclasses import dataclass

from qb_sim.engine.entities import PlayOutcome


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Coefficients for terminal outcomes, yardage, and time shaping."""

    completion: float = 1.0
    interception: float = -1.0
    incompletion: float = -0.2
    out_of_bounds: float = -0.2
    time_limit: float = -0.1
    yard: float = 0.02
    time_penalty: float = -0.001


def terminal_reward(outcome: PlayOutcome, yards: float, config: RewardConfig) -> float:
    """Return the terminal reward for a finished play."""
    if outcome is PlayOutcome.COMPLETED:
        return config.completion + yards * config.yard
    if outcome is PlayOutcome.INTERCEPTION:
        return config.interception
    if outcome is PlayOutcome.INCOMPLETE:
        return config.incompletion
    if outcome is PlayOutcome.OUT_OF_BOUNDS:
        return config.out_of_bounds
    if outcome is PlayOutcome.TIME_LIMIT:
        return config.time_limit
    return 0.0
