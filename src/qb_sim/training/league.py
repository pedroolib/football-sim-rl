"""Snapshot pools, league config and opponent sampling for MARL league rounds."""

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROLE_OPPONENTS: dict[str, tuple[str, ...]] = {
    "qb": ("wr1", "wr2", "cb1", "cb2", "ru1"),
    "wr": ("qb", "cb1", "cb2", "ru1"),
    "cb": ("qb", "wr1", "wr2", "ru1"),
    "ru": ("qb", "wr1", "wr2", "cb1", "cb2"),
}


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
