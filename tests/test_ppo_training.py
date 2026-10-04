import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402

from qb_sim.training.env import QBPassingEnv  # noqa: E402
from qb_sim.training.evaluation import evaluate  # noqa: E402
from qb_sim.training.ppo_config import PPOConfig  # noqa: E402
from qb_sim.training.train import train  # noqa: E402


def test_short_training_saves_model_config_and_checkpoints(tmp_path) -> None:
    config = PPOConfig(
        run_name="test-run",
        total_timesteps=512,
        seed=0,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        checkpoint_freq=256,
        eval_freq=256,
        eval_episodes=2,
        attempt_decay_steps=128,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    run_dir = tmp_path / "test-run"
    assert final_path.exists()
    assert (run_dir / "config.json").exists()
    assert list((run_dir / "checkpoints").glob("*.zip"))

    loaded = PPO.load(final_path, device="cpu")
    observation, _ = QBPassingEnv().reset(seed=0)
    action, _ = loaded.predict(observation, deterministic=True)
    assert np.asarray(action).shape == (6,)


def test_attempt_shaping_is_annealed_to_zero(tmp_path) -> None:
    config = PPOConfig(
        run_name="anneal-run",
        total_timesteps=256,
        seed=0,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=64,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )

    model, _ = train(config)

    assert model.get_env().envs[0].unwrapped.attempt_scale == 0.0


def test_training_warm_starts_from_demonstrations(tmp_path) -> None:
    config = PPOConfig(
        run_name="bc-run",
        scenario="rushed",
        total_timesteps=64,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=64,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    train(config)

    metadata = json.loads((tmp_path / "bc-run" / "config.json").read_text())
    assert metadata["imitation_samples"] > 0
    assert metadata["imitation_loss"] is not None


def test_ppo_action_policy_checks_layout_compatibility(tmp_path) -> None:
    config = PPOConfig(
        run_name="compat-run",
        scenario="rushed",
        total_timesteps=64,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )
    _, final_path = train(config)

    from qb_sim.training.evaluation import PPOActionPolicy

    policy = PPOActionPolicy(final_path, device="cpu")
    assert policy.supports_layout()


def test_training_can_cycle_scenarios(tmp_path) -> None:
    config = PPOConfig(
        run_name="multi-run",
        scenarios=("default", "rushed"),
        total_timesteps=64,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )

    train(config)

    metadata = json.loads((tmp_path / "multi-run" / "config.json").read_text())
    assert metadata["scenarios"] == ["default", "rushed"]


def test_ppo_evaluation_is_reproducible(tmp_path) -> None:
    config = PPOConfig(
        run_name="eval-run",
        total_timesteps=256,
        seed=1,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )
    model, _ = train(config)

    def policy(env, observation):
        action, _ = model.predict(observation, deterministic=True)
        return action

    first = evaluate(policy, "default", range(3))
    second = evaluate(policy, "default", range(3))

    assert first == second


def test_drive_training_smoke(tmp_path) -> None:
    from qb_sim.training.drive_observations import DRIVE_OBSERVATION_SIZE

    config = PPOConfig(
        run_name="drive-run",
        env_kind="drive",
        scenario="rushed",
        total_timesteps=256,
        seed=0,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=64,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER

    metadata = json.loads((tmp_path / "drive-run" / "config.json").read_text())
    assert metadata["env_kind"] == "drive"
    assert metadata["observation_size"] == DRIVE_OBSERVATION_SIZE
    assert metadata["route_pool"] == [route.value for route in ROUTE_OBSERVATION_ORDER]
    assert metadata["imitation_samples"] > 0

    loaded = PPO.load(final_path, device="cpu")
    assert loaded.observation_space.shape == (DRIVE_OBSERVATION_SIZE,)


def test_build_env_rejects_unknown_env_kind() -> None:
    from qb_sim.training.train import build_env

    with pytest.raises(ValueError):
        build_env(("rushed",), env_kind="driv")


def test_role_play_training_smoke(tmp_path) -> None:
    from qb_sim.training.role_context_observations import RECEIVER_CONTEXT_OBSERVATION_SIZE

    config = PPOConfig(
        run_name="role-play-run",
        env_kind="role_play",
        scenario="stage_b",
        learner="wr1",
        total_timesteps=128,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    metadata = json.loads((tmp_path / "role-play-run" / "config.json").read_text())
    assert metadata["env_kind"] == "role_play"
    assert metadata["learner"] == "wr1"
    assert metadata["observation_size"] == RECEIVER_CONTEXT_OBSERVATION_SIZE
    assert metadata["fatigue"]["carrier_fatigue_per_second"] > 0.0

    loaded = PPO.load(final_path, device="cpu")
    assert loaded.observation_space.shape == (RECEIVER_CONTEXT_OBSERVATION_SIZE,)


def test_role_drive_training_smoke(tmp_path) -> None:
    from qb_sim.training.role_context_observations import DEFENDER_CONTEXT_OBSERVATION_SIZE

    config = PPOConfig(
        run_name="role-drive-run",
        env_kind="role_drive",
        scenario="stage_b",
        learner="cb1",
        total_timesteps=128,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    metadata = json.loads((tmp_path / "role-drive-run" / "config.json").read_text())
    assert metadata["observation_size"] == DEFENDER_CONTEXT_OBSERVATION_SIZE
    assert metadata["learner"] == "cb1"


def test_training_with_two_subprocess_envs(tmp_path) -> None:
    config = PPOConfig(
        run_name="subproc-run",
        env_kind="role_play",
        scenario="stage_b",
        learner="wr1",
        total_timesteps=128,
        seed=0,
        n_envs=2,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )

    model, _ = train(config)

    metadata = json.loads((tmp_path / "subproc-run" / "config.json").read_text())
    assert metadata["n_envs"] == 2
    assert model.get_env().num_envs == 2


def test_annealing_callback_updates_subprocess_envs() -> None:
    from types import SimpleNamespace

    from qb_sim.training.train import AttemptAnnealingCallback, build_vec_env

    config = PPOConfig(env_kind="drive", scenario="stage_b", n_envs=2)
    env = build_vec_env(config, ("stage_b",))
    try:
        callback = AttemptAnnealingCallback(decay_steps=1)
        callback.model = SimpleNamespace(get_env=lambda: env)
        callback.num_timesteps = 10

        assert callback._on_step()
        assert env.get_attr("attempt_scale") == [0.0, 0.0]
    finally:
        env.close()


def test_role_play_qb_opponent_uses_the_drive_loader(monkeypatch) -> None:
    from qb_sim.training import policy
    from qb_sim.training.train import build_opponents

    calls: list[str] = []
    monkeypatch.setattr(
        policy,
        "load_policy",
        lambda path, name: calls.append("passing") or object(),
    )
    monkeypatch.setattr(
        policy,
        "load_drive_policy",
        lambda path, name: calls.append("drive") or object(),
    )

    build_opponents("role_play", "stage_b", (("qb", "model.zip"),))

    assert calls == ["drive"]


def test_randomized_role_play_training_records_ranges(tmp_path) -> None:
    config = PPOConfig(
        run_name="randomized-run",
        env_kind="role_play",
        scenario="stage_b",
        learner="wr1",
        randomize_config=True,
        n_envs=2,
        total_timesteps=128,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    metadata = json.loads((tmp_path / "randomized-run" / "config.json").read_text())
    assert metadata["randomize_config"] is True
    assert metadata["adaptation_ranges"]["receiver_speed"] == [6.0, 8.5]
    assert metadata["adaptation_ranges"]["defender_speed"] == [5.5, 7.5]
    assert metadata["adaptation_ranges"]["quarterback_speed"] == [5.0, 8.0]
    assert metadata["adaptation_ranges"]["rusher_speed"] == [5.0, 8.0]
    assert metadata["adaptation_ranges"]["throw_max_range"] == [30.0, 55.0]
    assert metadata["adaptation_ranges"]["throw_base_sigma_degrees"] == [0.5, 4.0]
    assert metadata["curriculum"]["decay_start_fraction"] == pytest.approx(0.4)
    assert metadata["curriculum"]["full_fraction"] == pytest.approx(0.8)
    assert final_path.exists()


@pytest.mark.parametrize("env_kind", ["role_play", "role_drive", "drive"])
def test_build_env_passes_randomization_ranges(env_kind) -> None:
    from qb_sim.training.randomization import AdaptationRanges
    from qb_sim.training.train import build_env

    learner = {"role_play": "wr1", "role_drive": "cb1", "drive": "qb"}[env_kind]
    env = build_env(("stage_b",), env_kind, learner, (), randomization=True)
    try:
        assert env.randomization == AdaptationRanges()
    finally:
        env.close()


def test_env_spec_forwards_randomization_to_build_env(monkeypatch) -> None:
    from qb_sim.rules.scenarios import get_scenario
    from qb_sim.training import train as train_module
    from qb_sim.training.role_env import RolePlayEnv

    captured: dict = {}

    def fake_build_env(
        scenario_names,
        env_kind="passing",
        learner="qb",
        opponents=(),
        randomization=None,
    ):
        captured["randomization"] = randomization
        return RolePlayEnv(get_scenario(scenario_names[0]), learner_id=learner)

    monkeypatch.setattr(train_module, "build_env", fake_build_env)

    env = train_module._EnvSpec(("stage_b",), "role_play", "wr1", (), True)()
    env.close()
    assert captured["randomization"] is True


def test_curriculum_callback_maps_progress_to_phase() -> None:
    from types import SimpleNamespace

    from qb_sim.training.train import AdaptationCurriculumCallback, build_vec_env

    config = PPOConfig(env_kind="role_play", scenario="stage_b", learner="wr1", n_envs=2)
    env = build_vec_env(config, ("stage_b",))
    try:
        callback = AdaptationCurriculumCallback()
        callback.model = SimpleNamespace(get_env=lambda: env)
        callback._total_timesteps = 100

        callback.num_timesteps = 20
        assert callback._on_step()
        assert env.get_attr("_curriculum_progress") == [0.0, 0.0]

        callback.num_timesteps = 60
        assert callback._on_step()
        assert env.get_attr("_curriculum_progress") == [
            pytest.approx(0.5),
            pytest.approx(0.5),
        ]

        callback.num_timesteps = 90
        assert callback._on_step()
        assert env.get_attr("_curriculum_progress") == [1.0, 1.0]
    finally:
        env.close()


def test_build_opponents_supports_rule_agents() -> None:
    from qb_sim.agents.rule_controllers import HeuristicQuarterback
    from qb_sim.training.train import build_opponents

    opponents = build_opponents("role_drive", "stage_b", (("qb", "rules"),))

    assert isinstance(opponents["qb"], HeuristicQuarterback)
