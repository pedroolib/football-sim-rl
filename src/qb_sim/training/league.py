"""Snapshot pools, league config and opponent sampling for MARL league rounds."""

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.evaluation import PPOActionPolicy, RuleQuarterbackPolicy
from qb_sim.training.policy import (
    load_defender_policy,
    load_drive_policy,
    load_receiver_policy,
)
from qb_sim.training.ppo_config import PPOConfig
from qb_sim.training.role_evaluation import run_composition_episode, summarize_composition

ROLE_OPPONENTS: dict[str, tuple[str, ...]] = {
    "qb": ("wr1", "wr2", "cb1", "cb2", "ru1"),
    "wr": ("qb", "cb1", "cb2", "ru1"),
    "cb": ("qb", "wr1", "wr2", "ru1"),
    "ru": ("qb", "wr1", "wr2", "cb1", "cb2"),
}

ROLE_ENVS: dict[str, tuple[str, str]] = {
    "qb": ("drive", "qb"),
    "wr": ("role_play", "wr1"),
    "cb": ("role_drive", "cb1"),
    "ru": ("role_drive", "ru1"),
}

LEAGUE_IMITATION_EPISODES = 300
LEAGUE_IMITATION_EPOCHS = 800
LEAGUE_IMITATION_LEARNING_RATE = 1e-4


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """Configuration for one league run."""

    rounds: int = 4
    steps: int = 150_000
    rotation: tuple[str, ...] = ("qb", "wr", "cb", "ru")
    pool_size: int = 3
    patience: int = 2
    scenario: str = "stage_b"
    seed: int = 0
    n_envs: int = 8
    n_steps: int = 128
    eval_episodes: int = 100
    eval_seed_start: int = 10_000
    output: str = "runs/league"

    def to_dict(self) -> dict:
        """Return a JSON-serializable copy of the configuration."""
        return dataclasses.asdict(self)


class SnapshotPool:
    """A directory of frozen policy snapshots, ordered oldest to newest."""

    def __init__(self, root: Path, pattern: str = "*.zip", pool_size: int = 3) -> None:
        self.root = Path(root)
        self.pattern = pattern
        self.pool_size = pool_size

    def files(self) -> list[Path]:
        """Return the snapshot files sorted by modification time, then name."""
        if not self.root.is_dir():
            return []
        found = [path for path in self.root.glob(self.pattern) if path.is_file()]
        return sorted(found, key=lambda path: (path.stat().st_mtime_ns, path.name))

    def latest(self) -> Path | None:
        """Return the newest snapshot, or None when the pool is empty."""
        files = self.files()
        return files[-1] if files else None

    def sample(self, rng: np.random.Generator) -> Path | None:
        """Return a deterministic random snapshot, or None when the pool is empty."""
        files = self.files()
        if not files:
            return None
        return files[int(rng.integers(len(files)))]

    def add(self, path: str | Path) -> Path:
        """Copy a snapshot into the pool, trimming the oldest beyond pool_size."""
        source = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        if source.resolve().parent == self.root.resolve():
            destination = source
        else:
            name = f"{source.parent.name}_{source.name}" if source.parent.name else source.name
            destination = self.root / name
            shutil.copy(source, destination)
        files = self.files()
        for stale in files[: max(0, len(files) - self.pool_size)]:
            stale.unlink()
        return destination


def opponent_spec(
    role: str,
    pools: dict[str, SnapshotPool],
    rng: np.random.Generator,
) -> tuple[tuple[str, str], ...]:
    """Sample one frozen snapshot per opponent player id for the trained role."""
    if role not in ROLE_OPPONENTS:
        raise ValueError(f"unknown league role: {role!r}")
    pairs: list[tuple[str, str]] = []
    for player_id in ROLE_OPPONENTS[role]:
        pool = pools.get(player_id)
        if pool is None:
            raise ValueError(f"no snapshot pool for opponent {player_id!r}")
        snapshot = pool.sample(rng)
        if snapshot is None:
            raise ValueError(f"empty snapshot pool for opponent {player_id!r}")
        pairs.append((player_id, str(snapshot)))
    return tuple(pairs)


def improved(points: float, previous: float | None, tolerance: float = 0.05) -> bool:
    """Return True when points beat the previous value by more than tolerance."""
    if previous is None:
        return True
    return points > previous + tolerance


def train_round(
    role: str,
    config: LeagueConfig,
    opponents: tuple[tuple[str, str], ...],
    round_index: int,
) -> Path:
    """Train one league round in-process and return the ``final_model.zip`` path."""
    if role not in ROLE_ENVS:
        raise ValueError(f"unknown league role: {role!r}")
    from qb_sim.training.train import train

    env_kind, learner = ROLE_ENVS[role]
    ppo_config = PPOConfig(
        run_name=f"league/r{round_index}_{role}",
        scenario=config.scenario,
        env_kind=env_kind,
        learner=learner,
        opponents=opponents,
        randomize_config=False,
        total_timesteps=config.steps,
        seed=config.seed + round_index,
        n_envs=config.n_envs,
        n_steps=config.n_steps,
        eval_episodes=config.eval_episodes,
        imitation_episodes=LEAGUE_IMITATION_EPISODES,
        imitation_epochs=LEAGUE_IMITATION_EPOCHS,
        imitation_learning_rate=LEAGUE_IMITATION_LEARNING_RATE,
        output_dir=config.output,
        device="auto",
    )
    (Path(config.output) / "league").mkdir(parents=True, exist_ok=True)
    _, final_path = train(ppo_config)
    return final_path


def load_role_policies(state: dict[str, str], scenario_name: str) -> dict[str, object]:
    """Load the current learned policy for every roster player."""
    return {
        "qb": load_drive_policy(state["qb"], scenario_name),
        "wr1": load_receiver_policy(state["wr"], scenario_name, "wr1"),
        "wr2": load_receiver_policy(state["wr"], scenario_name, "wr2"),
        "cb1": load_defender_policy(state["cb"], scenario_name, "cb1"),
        "cb2": load_defender_policy(state["cb"], scenario_name, "cb2"),
        "ru1": load_defender_policy(state["ru"], scenario_name, "ru1"),
    }


def evaluate_league(state: dict[str, str], config: LeagueConfig) -> dict:
    """Evaluate the learned composition and the rule baseline on the same seeds."""
    scenario = get_scenario(config.scenario)
    seeds = range(config.eval_seed_start, config.eval_seed_start + config.eval_episodes)
    policies = load_role_policies(state, config.scenario)
    learned_opponents = {
        player_id: policy for player_id, policy in policies.items() if player_id != "qb"
    }
    learned_env = QBDriveEnv(scenario, opponents=learned_opponents)
    qb_policy = PPOActionPolicy(state["qb"])
    learned_results = [run_composition_episode(learned_env, qb_policy, seed) for seed in seeds]
    learned_env.close()

    rules_env = QBDriveEnv(scenario, opponents={})
    rules_policy = RuleQuarterbackPolicy(scenario.config)
    rules_results = [run_composition_episode(rules_env, rules_policy, seed) for seed in seeds]
    rules_env.close()
    return {
        "learned": summarize_composition(learned_results),
        "rules": summarize_composition(rules_results),
    }


def league_row(
    round_index: int,
    role: str,
    opponents: tuple[tuple[str, str], ...],
    learned: dict[str, Any],
    rules: dict[str, Any],
    previous_points: float | None,
) -> dict:
    """Build one league table row with metrics and the improvement flag."""
    return {
        "round": round_index,
        "role": role,
        "opponents": dict(opponents),
        "learned": learned,
        "rules": rules,
        "improved": improved(learned["points_per_drive"], previous_points),
    }
