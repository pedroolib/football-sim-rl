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


@dataclass(frozen=True, slots=True)
class ReceiverRewardConfig:
    """Offense terminal reward plus a bonus when the learner catches."""

    play: RewardConfig = field(default_factory=RewardConfig)
    catch_bonus: float = 0.3
    touchdown_bonus: float = 7.0
    carry_yard: float = 0.05


def receiver_result_reward(
    outcome: PlayOutcome,
    yards: float,
    caught: bool,
    config: ReceiverRewardConfig,
) -> float:
    """Reward for one finished play from the receiver's perspective."""
    reward = terminal_reward(outcome, yards, config.play)
    if caught:
        reward += config.catch_bonus
    if outcome is PlayOutcome.TOUCHDOWN:
        reward += config.touchdown_bonus
    return float(reward)


@dataclass(frozen=True, slots=True)
class ReceiverDriveRewardConfig:
    """Team drive reward plus an openness bonus for the intended target."""

    receiver: ReceiverRewardConfig = field(default_factory=ReceiverRewardConfig)
    separation_bonus: float = 0.3
    separation_threshold: float = 3.0
    separation_shaping: bool = True
    shaping_gamma: float = 0.99
    shaping_scale: float = 5.0


def receiver_drive_reward(
    outcome: PlayOutcome,
    yards: float,
    caught: bool,
    targeted_open: bool,
    config: ReceiverDriveRewardConfig,
) -> float:
    """Reward for one finished drive play from the receiver's perspective.

    ``targeted_open`` means the learner was the intended receiver at the throw
    and its separation from the nearest coverage defender was at least the
    configured threshold.
    """
    reward = receiver_result_reward(outcome, yards, caught, config.receiver)
    if targeted_open:
        reward += config.separation_bonus
    return float(reward)


def receiver_separation_potential(separation: float, scale: float = 5.0) -> float:
    """Potential: more separation from the nearest defender is better (``[0, 1]``)."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return min(max(float(separation), 0.0), scale) / scale


def defender_drive_reward(result: PlayResult, config: DriveRewardConfig) -> float:
    """Zero-sum defender reward for one finished drive play."""
    return -drive_result_reward(result, config)


@dataclass(frozen=True, slots=True)
class DefenderRewardConfig:
    """Zero-sum terminal reward plus policy-invariant coverage shaping."""

    terminal: DriveRewardConfig = field(default_factory=DriveRewardConfig)
    shaping: bool = True
    shaping_gamma: float = 0.99
    shaping_scale: float = 10.0


def defender_potential(distance: float, scale: float = 10.0) -> float:
    """Potential: closer to the target is better (bounded to ``[-2, 0]``)."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return -min(max(float(distance), 0.0), 2.0 * scale) / scale
