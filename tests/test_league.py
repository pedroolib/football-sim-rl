import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

import qb_sim.training.league as league
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.league import (
    ROLE_OPPONENTS,
    LeagueConfig,
    SnapshotPool,
    evaluate_league,
    improved,
    league_row,
    load_role_policies,
    opponent_spec,
    train_round,
)

B2_MODELS = {
    "qb": Path("runs/qb/drive/adaptation_g1/final_model.zip"),
    "wr": Path("runs/wr/adaptation_g1/final_model.zip"),
    "cb": Path("runs/cb/adaptation_g1/final_model.zip"),
    "ru": Path("runs/ru/adaptation_g1/final_model.zip"),
}
B2_AVAILABLE = all(path.is_file() for path in B2_MODELS.values())


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
    assert config.device == "cpu"
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


def test_train_round_builds_role_specific_ppo_configs(monkeypatch, tmp_path: Path) -> None:
    captured = []

    def fake_train(ppo_config):
        captured.append(ppo_config)
        return None, Path(ppo_config.output_dir) / ppo_config.run_name / "final_model.zip"

    monkeypatch.setattr("qb_sim.training.train.train", fake_train)
    config = LeagueConfig(
        steps=1234,
        n_envs=3,
        n_steps=48,
        eval_episodes=7,
        seed=10,
        output=str(tmp_path),
    )
    opponents = (("qb", "snapshot.zip"),)

    expected = {
        "qb": ("drive", "qb"),
        "wr": ("role_play", "wr1"),
        "cb": ("role_drive", "cb1"),
        "ru": ("role_drive", "ru1"),
    }
    for role, (env_kind, learner) in expected.items():
        final_path = train_round(role, config, opponents, round_index=1)
        ppo_config = captured[-1]

        assert ppo_config.env_kind == env_kind
        assert ppo_config.learner == learner
        assert ppo_config.scenario == "stage_b"
        assert ppo_config.run_name == f"league/r1_{role}"
        assert ppo_config.total_timesteps == 1234
        assert ppo_config.opponents == opponents
        assert ppo_config.randomize_config is False
        assert ppo_config.imitation_episodes == 300
        assert ppo_config.imitation_epochs == 800
        assert ppo_config.imitation_learning_rate == 1e-4
        assert ppo_config.seed == 11
        assert ppo_config.device == "cpu"
        assert ppo_config.n_envs == 3
        assert ppo_config.n_steps == 48
        assert ppo_config.eval_episodes == 7
        assert ppo_config.output_dir == str(tmp_path)
        assert final_path == tmp_path / f"league/r1_{role}" / "final_model.zip"


def test_train_round_uses_the_config_device(monkeypatch, tmp_path: Path) -> None:
    captured = []

    def fake_train(ppo_config):
        captured.append(ppo_config)
        return None, Path(ppo_config.output_dir) / ppo_config.run_name / "final_model.zip"

    monkeypatch.setattr("qb_sim.training.train.train", fake_train)

    train_round(
        "wr",
        LeagueConfig(device="cuda:0", output=str(tmp_path)),
        (("qb", "rules"),),
        round_index=0,
    )

    assert captured[0].device == "cuda:0"


def test_train_round_rejects_unknown_role(tmp_path: Path) -> None:
    config = LeagueConfig(steps=1, output=str(tmp_path))

    with pytest.raises(ValueError, match="linebacker"):
        train_round("linebacker", config, (), round_index=0)


def test_train_round_smoke_trains_and_writes_config(tmp_path: Path) -> None:
    config = LeagueConfig(
        steps=256,
        n_envs=1,
        n_steps=64,
        eval_episodes=1,
        output=str(tmp_path),
    )
    opponents = (("qb", "rules"),)

    final_path = train_round("wr", config, opponents, round_index=0)

    assert final_path == tmp_path / "league" / "r0_wr" / "final_model.zip"
    assert final_path.is_file()
    metadata = json.loads((final_path.parent / "config.json").read_text(encoding="utf-8"))
    assert metadata["opponents"] == [["qb", "rules"]]
    assert metadata["env_kind"] == "role_play"
    assert metadata["learner"] == "wr1"
    assert metadata["seed"] == config.seed
    assert metadata["run_name"] == "league/r0_wr"


def test_load_role_policies_maps_state_paths_to_roster_players(monkeypatch) -> None:
    calls = []

    def fake_drive(path, scenario_name):
        calls.append(("drive", path, scenario_name))
        return f"drive:{path}"

    def fake_receiver(path, scenario_name, player_id):
        calls.append(("receiver", path, scenario_name, player_id))
        return f"receiver:{player_id}"

    def fake_defender(path, scenario_name, player_id):
        calls.append(("defender", path, scenario_name, player_id))
        return f"defender:{player_id}"

    monkeypatch.setattr(league, "load_drive_policy", fake_drive)
    monkeypatch.setattr(league, "load_receiver_policy", fake_receiver)
    monkeypatch.setattr(league, "load_defender_policy", fake_defender)

    state = {"qb": "qb.zip", "wr": "wr.zip", "cb": "cb.zip", "ru": "ru.zip"}
    policies = load_role_policies(state, "stage_b")

    assert policies == {
        "qb": "drive:qb.zip",
        "wr1": "receiver:wr1",
        "wr2": "receiver:wr2",
        "cb1": "defender:cb1",
        "cb2": "defender:cb2",
        "ru1": "defender:ru1",
    }
    assert ("drive", "qb.zip", "stage_b") in calls
    assert ("receiver", "wr.zip", "stage_b", "wr1") in calls
    assert ("receiver", "wr.zip", "stage_b", "wr2") in calls
    assert ("defender", "cb.zip", "stage_b", "cb1") in calls
    assert ("defender", "cb.zip", "stage_b", "cb2") in calls
    assert ("defender", "ru.zip", "stage_b", "ru1") in calls


def test_evaluate_league_runs_learned_and_rule_compositions(monkeypatch) -> None:
    envs = []
    runs = []
    summaries = []
    rule_configs = []

    class FakeEnv:
        def __init__(self, scenario, opponents=None):
            self.scenario = scenario
            self.opponents = opponents
            envs.append(self)

        def close(self):
            pass

    class FakePolicy:
        def __init__(self, path):
            self.path = path

    rule_policy = object()
    learned_policies = {
        "qb": object(),
        "wr1": object(),
        "wr2": object(),
        "cb1": object(),
        "cb2": object(),
        "ru1": object(),
    }
    monkeypatch.setattr(league, "QBDriveEnv", FakeEnv)
    monkeypatch.setattr(league, "PPOActionPolicy", FakePolicy)
    monkeypatch.setattr(
        league,
        "RuleQuarterbackPolicy",
        lambda config: rule_configs.append(config) or rule_policy,
    )
    monkeypatch.setattr(league, "load_role_policies", lambda state, name: dict(learned_policies))
    monkeypatch.setattr(
        league,
        "run_composition_episode",
        lambda env, policy, seed: runs.append((env, policy, seed)) or {"fake": True},
    )
    monkeypatch.setattr(
        league,
        "summarize_composition",
        lambda results: summaries.append(list(results)) or {"points_per_drive": 1.0},
    )

    state = {"qb": "qb.zip", "wr": "wr.zip", "cb": "cb.zip", "ru": "ru.zip"}
    report = evaluate_league(state, LeagueConfig(eval_episodes=2, eval_seed_start=500))

    assert report == {
        "learned": {"points_per_drive": 1.0},
        "rules": {"points_per_drive": 1.0},
    }
    assert [seed for _, _, seed in runs] == [500, 501, 500, 501]
    assert runs[0][1].path == "qb.zip"
    assert runs[2][1] is rule_policy
    assert runs[3][1] is rule_policy
    assert envs[0].scenario.name == "stage_b"
    assert set(envs[0].opponents) == {"wr1", "wr2", "cb1", "cb2", "ru1"}
    assert envs[0].opponents["wr1"] is learned_policies["wr1"]
    assert envs[1].opponents == {}
    assert rule_configs == [get_scenario("stage_b").config]
    assert [len(batch) for batch in summaries] == [2, 2]


def test_evaluate_league_reuses_one_qb_policy_per_call(monkeypatch) -> None:
    created = []

    class CountingPolicy:
        def __init__(self, path):
            created.append(path)

        def __call__(self, env, observation):
            return np.zeros(6, dtype=np.float32)

    class FakeEnv:
        def __init__(self, scenario, opponents=None):
            pass

        def close(self):
            pass

    monkeypatch.setattr(league, "PPOActionPolicy", CountingPolicy)
    monkeypatch.setattr(league, "QBDriveEnv", FakeEnv)
    monkeypatch.setattr(league, "RuleQuarterbackPolicy", lambda config: object())
    monkeypatch.setattr(league, "load_role_policies", lambda state, name: {})
    monkeypatch.setattr(league, "run_composition_episode", lambda env, policy, seed: {"fake": True})
    monkeypatch.setattr(league, "summarize_composition", lambda results: {"points_per_drive": 1.0})

    first = {"qb": "first.zip", "wr": "wr.zip", "cb": "cb.zip", "ru": "ru.zip"}
    second = {"qb": "second.zip", "wr": "wr.zip", "cb": "cb.zip", "ru": "ru.zip"}

    evaluate_league(first, LeagueConfig(eval_episodes=3, eval_seed_start=0))
    evaluate_league(second, LeagueConfig(eval_episodes=3, eval_seed_start=0))

    assert created == ["first.zip", "second.zip"]


def test_league_row_records_metrics_and_improvement() -> None:
    learned = {"points_per_drive": 2.0, "touchdown_pct": 30.0}
    rules = {"points_per_drive": 1.5, "touchdown_pct": 20.0}
    opponents = (("wr1", "wr_snap.zip"), ("cb1", "cb_snap.zip"))

    row = league_row(3, "qb", opponents, learned, rules, previous_points=None)

    assert row["round"] == 3
    assert row["role"] == "qb"
    assert row["opponents"] == {"wr1": "wr_snap.zip", "cb1": "cb_snap.zip"}
    assert row["learned"] == learned
    assert row["rules"] == rules
    assert row["improved"] is True

    inside = league_row(3, "qb", opponents, learned, rules, previous_points=2.0)
    beyond = league_row(3, "qb", opponents, learned, rules, previous_points=1.90)
    assert inside["improved"] is False
    assert beyond["improved"] is True


@pytest.mark.skipif(not B2_AVAILABLE, reason="B2 generalized models are not present")
def test_evaluate_league_with_the_b2_models() -> None:
    state = {role: str(path) for role, path in B2_MODELS.items()}

    report = evaluate_league(state, LeagueConfig(eval_episodes=2, eval_seed_start=10_000))

    assert set(report) == {"learned", "rules"}
    for summary in report.values():
        assert summary["episodes"] == 2.0
        assert summary["points_per_drive"] is not None
        assert summary["touchdown_pct"] in {0.0, 50.0, 100.0}

    row = league_row(0, "qb", (), report["learned"], report["rules"], previous_points=None)
    assert row["improved"] is True


def _load_marl_league():
    path = Path(__file__).resolve().parents[1] / "scripts" / "marl_league.py"
    spec = importlib.util.spec_from_file_location("marl_league", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_json(tmp_path: Path) -> Path:
    if B2_AVAILABLE:
        base = {role: str(path) for role, path in B2_MODELS.items()}
    else:
        base = {}
        for role in ("qb", "wr", "cb", "ru"):
            model = tmp_path / f"base_{role}.zip"
            model.write_bytes(b"model")
            base[role] = str(model)
    path = tmp_path / "base.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


def _fake_train_round(role, config, opponents, round_index) -> Path:
    final = Path(config.output) / f"league/r{round_index}_{role}/final_model.zip"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"model")
    return final


def test_marl_league_smoke_writes_state_and_table(tmp_path: Path, monkeypatch) -> None:
    marl_league = _load_marl_league()
    base_json = _base_json(tmp_path)
    evaluated_states = []

    def fake_evaluate_league(state, config):
        evaluated_states.append(dict(state))
        return {
            "learned": {"points_per_drive": 2.0, "touchdown_pct": 50.0},
            "rules": {"points_per_drive": 1.0, "touchdown_pct": 25.0},
        }

    monkeypatch.setattr(marl_league, "train_round", _fake_train_round)
    monkeypatch.setattr(marl_league, "evaluate_league", fake_evaluate_league)

    result = marl_league.main(
        [
            "--rounds",
            "1",
            "--steps",
            "256",
            "--n-envs",
            "1",
            "--n-steps",
            "64",
            "--eval-episodes",
            "1",
            "--output",
            str(tmp_path),
            "--base",
            str(base_json),
        ]
    )

    state_path = tmp_path / "league_state.json"
    table_path = tmp_path / "league_table.json"
    assert result == 0
    assert state_path.is_file()
    assert table_path.is_file()
    assert (tmp_path / "league_table.csv").is_file()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = json.loads(table_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    assert row["round"] == 0
    assert row["role"] == "qb"
    assert set(row["opponents"]) == {"wr1", "wr2", "cb1", "cb2", "ru1"}
    assert row["learned"] == {"points_per_drive": 2.0, "touchdown_pct": 50.0}
    assert row["rules"] == {"points_per_drive": 1.0, "touchdown_pct": 25.0}
    assert row["improved"] is True

    assert state["round"] == 1
    assert state["table"] == str(table_path)
    assert state["models"]["qb"] == str(tmp_path / "league" / "r0_qb" / "final_model.zip")
    assert state["models"]["wr"] == json.loads(base_json.read_text(encoding="utf-8"))["wr"]
    assert set(evaluated_states[0]) == {"qb", "wr", "cb", "ru"}
    assert evaluated_states[0]["qb"] == state["models"]["qb"]


def test_marl_league_passes_device_to_train_round(tmp_path: Path, monkeypatch) -> None:
    marl_league = _load_marl_league()
    base_json = _base_json(tmp_path)
    devices = []

    def fake_train_round(role, config, opponents, round_index):
        devices.append(config.device)
        return _fake_train_round(role, config, opponents, round_index)

    def fake_evaluate_league(state, config):
        return {"learned": {"points_per_drive": 1.0}, "rules": {"points_per_drive": 1.0}}

    monkeypatch.setattr(marl_league, "train_round", fake_train_round)
    monkeypatch.setattr(marl_league, "evaluate_league", fake_evaluate_league)

    marl_league.main(
        [
            "--rounds",
            "1",
            "--device",
            "cuda:0",
            "--n-envs",
            "1",
            "--n-steps",
            "64",
            "--eval-episodes",
            "1",
            "--output",
            str(tmp_path),
            "--base",
            str(base_json),
        ]
    )

    assert devices == ["cuda:0"]


def test_marl_league_forces_cpu_fork_safe_env(tmp_path: Path, monkeypatch) -> None:
    marl_league = _load_marl_league()
    base_json = _base_json(tmp_path)
    observed = []

    def fake_train_round(role, config, opponents, round_index):
        observed.append((os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("OMP_NUM_THREADS")))
        return _fake_train_round(role, config, opponents, round_index)

    def fake_evaluate_league(state, config):
        observed.append((os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("OMP_NUM_THREADS")))
        return {"learned": {"points_per_drive": 1.0}, "rules": {"points_per_drive": 1.0}}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setattr(marl_league, "train_round", fake_train_round)
    monkeypatch.setattr(marl_league, "evaluate_league", fake_evaluate_league)

    marl_league.main(
        [
            "--rounds",
            "1",
            "--n-envs",
            "1",
            "--n-steps",
            "64",
            "--eval-episodes",
            "1",
            "--output",
            str(tmp_path),
            "--base",
            str(base_json),
        ]
    )

    assert observed == [("", "1"), ("", "1")]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["OMP_NUM_THREADS"] == "4"


def test_marl_league_stops_after_patience_without_improvement(tmp_path: Path, monkeypatch) -> None:
    marl_league = _load_marl_league()
    base_json = _base_json(tmp_path)
    trained = []

    def fake_train_round(role, config, opponents, round_index) -> Path:
        trained.append((round_index, role))
        return _fake_train_round(role, config, opponents, round_index)

    def fake_evaluate_league(state, config):
        return {"learned": {"points_per_drive": 1.0}, "rules": {"points_per_drive": 1.0}}

    monkeypatch.setattr(marl_league, "train_round", fake_train_round)
    monkeypatch.setattr(marl_league, "evaluate_league", fake_evaluate_league)

    marl_league.main(
        [
            "--rounds",
            "10",
            "--steps",
            "256",
            "--n-envs",
            "1",
            "--n-steps",
            "64",
            "--eval-episodes",
            "1",
            "--output",
            str(tmp_path),
            "--base",
            str(base_json),
        ]
    )

    rows = json.loads((tmp_path / "league_table.json").read_text(encoding="utf-8"))
    assert [row["improved"] for row in rows] == [True, False, False]
    assert [entry[1] for entry in trained] == ["qb", "wr", "cb"]
    state = json.loads((tmp_path / "league_state.json").read_text(encoding="utf-8"))
    assert state["round"] == 3


def test_marl_league_resumes_from_existing_state(tmp_path: Path, monkeypatch) -> None:
    marl_league = _load_marl_league()
    base_json = _base_json(tmp_path)

    def fake_evaluate_league(state, config):
        return {"learned": {"points_per_drive": 2.0}, "rules": {"points_per_drive": 1.0}}

    monkeypatch.setattr(marl_league, "train_round", _fake_train_round)
    monkeypatch.setattr(marl_league, "evaluate_league", fake_evaluate_league)
    common = ["--n-envs", "1", "--n-steps", "64", "--eval-episodes", "1", "--output", str(tmp_path)]

    marl_league.main(["--rounds", "1", "--base", str(base_json), *common])
    marl_league.main(["--rounds", "2", "--resume", str(tmp_path / "league_state.json"), *common])

    rows = json.loads((tmp_path / "league_table.json").read_text(encoding="utf-8"))
    assert [row["role"] for row in rows] == ["qb", "wr"]
    state = json.loads((tmp_path / "league_state.json").read_text(encoding="utf-8"))
    assert state["round"] == 2


def test_marl_league_rejects_invalid_base_before_training(tmp_path: Path, monkeypatch) -> None:
    marl_league = _load_marl_league()

    def fail_train_round(*args, **kwargs):
        raise AssertionError("training must not start with an invalid base")

    monkeypatch.setattr(marl_league, "train_round", fail_train_round)
    rules_base = tmp_path / "rules_base.json"
    rules_base.write_text(json.dumps({role: "rules" for role in ("qb", "wr", "cb", "ru")}))
    missing_base = tmp_path / "missing_base.json"
    missing_base.write_text(
        json.dumps(
            {role: str(tmp_path / f"missing_{role}.zip") for role in ("qb", "wr", "cb", "ru")}
        )
    )

    for base_json in (rules_base, missing_base):
        with pytest.raises(SystemExit):
            marl_league.main(
                [
                    "--rounds",
                    "1",
                    "--n-envs",
                    "1",
                    "--n-steps",
                    "64",
                    "--eval-episodes",
                    "1",
                    "--output",
                    str(tmp_path),
                    "--base",
                    str(base_json),
                ]
            )

    assert not (tmp_path / "league").exists()


def test_marl_league_rejects_unknown_rotation_role(tmp_path: Path, monkeypatch) -> None:
    marl_league = _load_marl_league()
    base_json = _base_json(tmp_path)

    def fail_train_round(*args, **kwargs):
        raise AssertionError("training must not start with an unknown rotation role")

    monkeypatch.setattr(marl_league, "train_round", fail_train_round)

    with pytest.raises(SystemExit):
        marl_league.main(
            [
                "--rotation",
                "qb,linebacker",
                "--rounds",
                "1",
                "--output",
                str(tmp_path),
                "--base",
                str(base_json),
            ]
        )

    assert not (tmp_path / "league").exists()


def test_marl_league_restores_stalled_on_resume(tmp_path: Path, monkeypatch) -> None:
    marl_league = _load_marl_league()
    base_json = _base_json(tmp_path)

    def fake_evaluate_league(state, config):
        return {"learned": {"points_per_drive": 1.0}, "rules": {"points_per_drive": 1.0}}

    monkeypatch.setattr(marl_league, "train_round", _fake_train_round)
    monkeypatch.setattr(marl_league, "evaluate_league", fake_evaluate_league)
    common = ["--n-envs", "1", "--n-steps", "64", "--eval-episodes", "1", "--output", str(tmp_path)]

    marl_league.main(["--rounds", "2", "--base", str(base_json), *common])
    state_path = tmp_path / "league_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["stalled"] == 1

    marl_league.main(["--rounds", "10", "--resume", str(state_path), *common])

    rows = json.loads((tmp_path / "league_table.json").read_text(encoding="utf-8"))
    assert [row["improved"] for row in rows] == [True, False, False]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["stalled"] == 2
    assert state["round"] == 3
