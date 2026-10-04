import os
from pathlib import Path

import numpy as np
import pytest

from qb_sim.training.league import (
    ROLE_OPPONENTS,
    LeagueConfig,
    SnapshotPool,
    improved,
    opponent_spec,
)


def _write_snapshot(path: Path, mtime_ns: int) -> Path:
    path.write_bytes(b"snapshot")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _make_pool(tmp_path: Path, player: str, count: int = 3) -> SnapshotPool:
    root = tmp_path / player
    root.mkdir()
    for index in range(count):
        _write_snapshot(root / f"{player}_{index}.zip", 1_000 + index)
    return SnapshotPool(root, pool_size=count)


def _all_player_ids() -> list[str]:
    return sorted({player for ids in ROLE_OPPONENTS.values() for player in ids})


def test_snapshot_pool_discovers_and_sorts_by_mtime_then_name(tmp_path: Path) -> None:
    root = tmp_path / "pool"
    root.mkdir()
    _write_snapshot(root / "b.zip", 2_000)
    _write_snapshot(root / "a.zip", 1_000)
    _write_snapshot(root / "c.zip", 1_000)
    (root / "notes.txt").write_text("ignore")
    (root / "folder.zip").mkdir()

    pool = SnapshotPool(root)

    assert pool.files() == [root / "a.zip", root / "c.zip", root / "b.zip"]


def test_snapshot_pool_latest_returns_newest_or_none(tmp_path: Path) -> None:
    pool = SnapshotPool(tmp_path / "pool")
    assert pool.files() == []
    assert pool.latest() is None

    root = tmp_path / "pool"
    root.mkdir()
    _write_snapshot(root / "old.zip", 1_000)
    newest = _write_snapshot(root / "new.zip", 2_000)

    assert SnapshotPool(root).latest() == newest


def test_snapshot_pool_sample_is_deterministic_and_none_when_empty(tmp_path: Path) -> None:
    root = tmp_path / "pool"
    root.mkdir()
    for index in range(4):
        _write_snapshot(root / f"model_{index}.zip", 1_000 + index)

    pool = SnapshotPool(root)
    first = pool.sample(np.random.default_rng(7))
    second = pool.sample(np.random.default_rng(7))

    assert first == second
    assert first in pool.files()
    assert SnapshotPool(tmp_path / "missing").sample(np.random.default_rng(0)) is None


def test_snapshot_pool_add_copies_and_trims_keeping_newest(tmp_path: Path) -> None:
    pool = SnapshotPool(tmp_path / "pool", pool_size=2)
    source = tmp_path / "src"
    source.mkdir()

    for index in range(4):
        snapshot = source / f"snap{index}.zip"
        snapshot.write_bytes(f"snap{index}".encode())
        pool.add(snapshot)

    files = pool.files()
    assert [path.name for path in files] == ["src_snap2.zip", "src_snap3.zip"]
    assert files[-1].read_bytes() == b"snap3"


def test_league_config_defaults_match_the_spec() -> None:
    config = LeagueConfig()

    assert config.rounds == 4
    assert config.steps == 150_000
    assert config.rotation == ("qb", "wr", "cb", "ru")
    assert config.pool_size == 3
    assert config.patience == 2
    assert config.scenario == "stage_b"
    assert config.seed == 0
    assert config.n_envs == 8
    assert config.n_steps == 128
    assert config.eval_episodes == 100
    assert config.eval_seed_start == 10_000
    assert config.output == "runs/league"


def test_role_opponents_mapping() -> None:
    assert ROLE_OPPONENTS == {
        "qb": ("wr1", "wr2", "cb1", "cb2", "ru1"),
        "wr": ("qb", "cb1", "cb2", "ru1"),
        "cb": ("qb", "wr1", "wr2", "ru1"),
        "ru": ("qb", "wr1", "wr2", "cb1", "cb2"),
    }


def test_opponent_spec_samples_one_snapshot_per_opponent(tmp_path: Path) -> None:
    pools = {player: _make_pool(tmp_path, player) for player in _all_player_ids()}

    spec = opponent_spec("wr", pools, np.random.default_rng(3))

    assert [player for player, _ in spec] == list(ROLE_OPPONENTS["wr"])
    assert all(Path(path) in pools[player].files() for player, path in spec)


def test_opponent_spec_is_deterministic_for_the_same_seed(tmp_path: Path) -> None:
    pools = {player: _make_pool(tmp_path, player) for player in _all_player_ids()}

    first = opponent_spec("cb", pools, np.random.default_rng(11))
    second = opponent_spec("cb", pools, np.random.default_rng(11))

    assert first == second


def test_opponent_spec_raises_when_required_pool_is_empty(tmp_path: Path) -> None:
    pools = {player: _make_pool(tmp_path, player) for player in _all_player_ids()}
    pools["cb1"] = SnapshotPool(tmp_path / "empty_cb1")

    with pytest.raises(ValueError, match="cb1"):
        opponent_spec("qb", pools, np.random.default_rng(0))


def test_opponent_spec_raises_when_required_pool_is_missing(tmp_path: Path) -> None:
    pools = {player: _make_pool(tmp_path, player) for player in _all_player_ids()}
    del pools["ru1"]

    with pytest.raises(ValueError, match="ru1"):
        opponent_spec("qb", pools, np.random.default_rng(0))


def test_improved_uses_the_tolerance_band() -> None:
    assert improved(2.0, None) is True
    assert improved(1.10, 1.00) is True
    assert improved(1.06, 1.00) is True
    assert improved(1.05, 1.00) is False
    assert improved(1.03, 1.00) is False
    assert improved(0.90, 1.00) is False
    assert improved(1.01, 1.00, tolerance=0.0) is True
