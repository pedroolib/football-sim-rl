from dataclasses import replace

import pytest

from qb_sim.agents.rule_controllers import HeuristicQuarterback
from qb_sim.engine.actions import Throw
from qb_sim.engine.entities import BallStatus
from qb_sim.engine.geometry import vec2
from qb_sim.rules.drive import DriveConfig
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.monitoring import (
    CompositionDemoPanel,
    DemoPanel,
    DriveDemoPanel,
    RoleDriveDemoPanel,
    RolePlayDemoPanel,
    checkpoint_catalog,
    checkpoint_label,
    checkpoint_steps,
    latest_checkpoint,
    read_progress,
    run_has_training_artifacts,
    total_timesteps,
)


def test_checkpoint_steps_parses_the_filename() -> None:
    assert checkpoint_steps("ppo_default_50000_steps.zip") == 50000
    assert checkpoint_steps("weird.zip") == -1


def test_latest_checkpoint_picks_the_highest_step_count(tmp_path) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "run_50000_steps.zip").touch()
    (directory / "run_100000_steps.zip").touch()

    assert latest_checkpoint(tmp_path).name == "run_100000_steps.zip"


def test_latest_checkpoint_returns_none_without_checkpoints(tmp_path) -> None:
    assert latest_checkpoint(tmp_path) is None


def test_read_progress_parses_csv(tmp_path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "progress.csv").write_text(
        "time/total_timesteps,rollout/ep_rew_mean\n1024,-0.24\n2048,-0.20\n",
        encoding="utf-8",
    )

    assert read_progress(tmp_path) == [(1024, -0.24), (2048, -0.20)]


def test_demo_panel_finishes_and_restarts() -> None:
    panel = DemoPanel(Play(), policy=None)

    for _ in range(200):
        panel.tick()
        if panel.finished:
            break

    assert panel.finished >= 1
    assert panel.last_outcome != ""

    for _ in range(50):
        panel.tick()

    assert panel.finished >= 1


def test_checkpoint_catalog_orders_steps_then_extras(tmp_path) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "run_100000_steps.zip").touch()
    (directory / "run_50000_steps.zip").touch()
    (tmp_path / "best").mkdir()
    (tmp_path / "best" / "best_model.zip").touch()
    (tmp_path / "final_model.zip").touch()

    catalog = checkpoint_catalog(tmp_path)

    assert [path.name for path in catalog] == [
        "run_50000_steps.zip",
        "run_100000_steps.zip",
        "best_model.zip",
        "final_model.zip",
    ]
    assert checkpoint_label(catalog[0]) == "step 50,000"
    assert checkpoint_label(catalog[2]) == "best"
    assert checkpoint_label(catalog[3]) == "final"


def test_checkpoint_catalog_is_empty_without_files(tmp_path) -> None:
    assert checkpoint_catalog(tmp_path) == []


def test_total_timesteps_reads_the_run_config(tmp_path) -> None:
    (tmp_path / "config.json").write_text('{"total_timesteps": 250000}', encoding="utf-8")

    assert total_timesteps(tmp_path) == 250000
    assert total_timesteps(tmp_path / "missing") is None


def test_drive_panel_plays_drives_and_tracks_stats() -> None:
    panel = DriveDemoPanel(
        get_scenario("rushed"),
        policy=None,
        seed=0,
        drive_config=DriveConfig(los_range=(40.0, 40.0)),
    )

    for _ in range(600):
        panel.tick()
        if panel.drives >= 1:
            break

    assert panel.drives >= 1
    assert panel.last_outcome != ""

    for _ in range(20):
        panel.tick()

    assert panel.down_label.startswith(("1st", "2nd", "3rd"))


def test_drive_panel_is_deterministic_for_a_seed() -> None:
    config = DriveConfig(los_range=(40.0, 40.0))
    first = DriveDemoPanel(get_scenario("rushed"), seed=7, drive_config=config)
    second = DriveDemoPanel(get_scenario("rushed"), seed=7, drive_config=config)

    assert first.drive.routes == second.drive.routes
    assert first.drive.state.los == second.drive.state.los


def _rule_qb() -> HeuristicQuarterback:
    return HeuristicQuarterback("qb", get_scenario("stage_b").config)


def test_role_play_panel_runs_plays_with_rule_agents() -> None:
    panel = RolePlayDemoPanel(
        get_scenario("stage_b"),
        learner_id="wr1",
        opponents={"qb": _rule_qb()},
        seed=0,
    )

    for _ in range(600):
        panel.tick()
        if panel.plays >= 1:
            break

    assert panel.plays >= 1
    assert panel.last_outcome != ""


def test_role_play_panel_is_deterministic_for_a_seed() -> None:
    scenario = get_scenario("stage_b")
    first = RolePlayDemoPanel(scenario, seed=7, opponents={"qb": _rule_qb()})
    second = RolePlayDemoPanel(scenario, seed=7, opponents={"qb": _rule_qb()})

    for _ in range(30):
        first.tick()
        second.tick()

    assert first.play.state.player("wr1").position.tolist() == pytest.approx(
        second.play.state.player("wr1").position.tolist()
    )
    assert first.routes == second.routes
    assert first.play.line_of_scrimmage == pytest.approx(second.play.line_of_scrimmage)


def test_role_play_panel_calls_the_qb_opponent_and_throws() -> None:
    calls: list[int] = []

    def qb(state, routes, drive_state):
        calls.append(1)
        if state.ball.status is BallStatus.HELD:
            return Throw(direction=vec2(1.0, 0.0), power=1.0)
        return None

    panel = RolePlayDemoPanel(
        get_scenario("stage_b"),
        learner_id="wr1",
        opponents={"qb": qb},
        seed=0,
    )

    for _ in range(5):
        panel.tick()

    assert calls
    assert panel.play.state.throw_time is not None


def test_role_drive_panel_calls_the_qb_opponent() -> None:
    calls: list[int] = []

    def qb(state, routes, drive_state):
        calls.append(1)
        return None

    panel = RoleDriveDemoPanel(
        get_scenario("stage_b"),
        learner_id="cb1",
        opponents={"qb": qb},
        seed=0,
        drive_config=DriveConfig(los_range=(40.0, 40.0)),
    )

    panel.tick()

    assert calls


def test_run_has_training_artifacts(tmp_path) -> None:
    assert not run_has_training_artifacts(tmp_path)

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert run_has_training_artifacts(tmp_path)


def test_role_play_panel_samples_the_los_inside_the_range() -> None:
    contexts: list[float] = []

    def qb(state, routes, drive_state):
        contexts.append(drive_state.los)
        return None

    panel = RolePlayDemoPanel(
        get_scenario("stage_b"),
        learner_id="wr1",
        opponents={"qb": qb},
        seed=0,
        los_range=(30.0, 40.0),
    )
    panel.tick()

    los = panel.play.line_of_scrimmage
    assert 30.0 <= los <= 40.0
    assert panel.play.state.player("qb").position[0] == pytest.approx(los)
    assert contexts[0] == pytest.approx(los)


def test_role_play_panel_passes_the_shifted_start_to_the_policy() -> None:
    starts: list[tuple[float, float]] = []

    def policy(state, route, start):
        starts.append(tuple(start))
        return None

    panel = RolePlayDemoPanel(
        get_scenario("stage_b"),
        learner_id="wr1",
        opponents={"qb": _rule_qb()},
        policy=policy,
        seed=0,
        los_range=(45.0, 45.0),
    )
    panel.tick()

    assert starts
    assert starts[0][0] == pytest.approx(45.0)
    assert starts[0][1] == pytest.approx(43.0)


def test_role_play_panel_measures_yards_from_the_los() -> None:
    panel = RolePlayDemoPanel(
        get_scenario("stage_b"),
        learner_id="wr1",
        opponents={"qb": _rule_qb()},
        seed=0,
        los_range=(40.0, 40.0),
    )
    state = panel.play.state
    state.player("wr1").position = vec2(52.0, 43.0)
    state.carrier_id = "wr1"
    state.ball.status = BallStatus.CARRIED
    state.carry_started_at = state.elapsed_seconds

    for _ in range(600):
        panel.tick()
        if panel.plays >= 1:
            break

    assert panel.plays >= 1
    expected = panel.play.state.player("wr1").position[0] - panel.play.line_of_scrimmage
    assert panel.yards == pytest.approx(expected)


def test_role_drive_panel_counts_sacks() -> None:
    def passive_qb(state, routes, drive_state):
        return None

    panel = RoleDriveDemoPanel(
        get_scenario("stage_b"),
        learner_id="cb1",
        opponents={"qb": passive_qb},
        seed=0,
        drive_config=DriveConfig(los_range=(40.0, 40.0)),
    )

    for _ in range(2000):
        panel.tick()
        if panel.drives >= 1:
            break

    assert panel.drives >= 1
    assert panel.sacks >= 1


def test_composition_panel_runs_drives_with_a_qb_policy() -> None:
    calls: list[int] = []

    def qb(state, routes, drive_state):
        calls.append(1)
        return None

    panel = CompositionDemoPanel(
        get_scenario("stage_b"),
        qb_policy=qb,
        seed=0,
        drive_config=DriveConfig(los_range=(40.0, 40.0)),
    )

    for _ in range(3000):
        panel.tick()
        if panel.drives >= 1:
            break

    assert calls
    assert panel.drives >= 1
    assert panel.last_outcome != ""


def test_composition_panel_rebuild_uses_the_new_config() -> None:
    scenario = get_scenario("stage_b")
    panel = CompositionDemoPanel(
        scenario,
        seed=0,
        drive_config=DriveConfig(los_range=(40.0, 40.0)),
    )

    faster = replace(scenario, config=replace(scenario.config, receiver_speed=8.5))
    panel.rebuild(faster, DriveConfig(los_range=(40.0, 40.0)), None, {})

    assert panel.play.config.receiver_speed == 8.5


def test_role_drive_panel_tracks_drives() -> None:
    scenario = get_scenario("stage_b")
    panel = RoleDriveDemoPanel(
        scenario,
        learner_id="cb1",
        opponents={"qb": _rule_qb()},
        seed=0,
        drive_config=DriveConfig(los_range=(40.0, 40.0)),
    )

    for _ in range(2000):
        panel.tick()
        if panel.drives >= 1:
            break

    assert panel.drives >= 1
    assert panel.last_outcome != ""
    assert panel.down_label != ""
