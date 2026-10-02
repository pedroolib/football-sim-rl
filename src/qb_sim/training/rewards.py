"""Configurable reward function for the quarterback environment."""

from dataclasses import dataclass, field

from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.drive import PlayResult


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Coefficients for terminal outcomes, yardage, and time shaping."""

    completion: float = 1.0
    interception: float = -1.0
    incompletion: float = -0.2
    out_of_bounds: float = -0.2
    time_limit: float = -0.5
    sack: float = -1.0
    yard: float = 0.02
    time_penalty: float = -0.002
    attempt: float = 0.1
    attempt_min_seconds: float = 0.5


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
    if outcome is PlayOutcome.SACK:
        return config.sack
    return 0.0


@dataclass(frozen=True, slots=True)
class DriveRewardConfig:
    """Per-play drive reward with first-down and turnover terms."""

    play: RewardConfig = field(default_factory=RewardConfig)
    first_down: float = 0.3
    turnover_on_downs: float = -0.5


def drive_result_reward(result: PlayResult, config: DriveRewardConfig) -> float:
    """Return the reward for one finished drive play."""
    reward = terminal_reward(result.outcome, result.yards, config.play)
    if result.first_down:
        reward += config.first_down
    if result.touchdown:
        reward += result.points
    if result.end_reason == "turnover_on_downs":
        reward += config.turnover_on_downs
    return float(reward)
