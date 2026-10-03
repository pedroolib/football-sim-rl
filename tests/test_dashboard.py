import pytest

pygame = pytest.importorskip("pygame")

from qb_sim.agents.rule_controllers import HeuristicQuarterback  # noqa: E402
from qb_sim.rules.scenarios import get_scenario  # noqa: E402
from qb_sim.ui.dashboard import TrainingDashboard  # noqa: E402


class _StubPolicy:
    def __call__(self, state):
        return None


class _StubRolePolicy:
    def __call__(self, *args):
        return None


def _rule_qb() -> HeuristicQuarterback:
    return HeuristicQuarterback("qb", get_scenario("stage_b").config)


@pytest.fixture
def dummy_video(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")


def test_dashboard_steps_and_draws(dummy_video, tmp_path) -> None:
    dashboard = TrainingDashboard(
        tmp_path,
        panels=2,
        screen_size=(800, 450),
        policy_factory=lambda path: _StubPolicy(),
    )
    try:
        for _ in range(40):
            dashboard.step_frames(0.05)
        dashboard.draw_frame()

        assert all(panel.play.state.elapsed_seconds > 0.0 for panel in dashboard.panels)
    finally:
        dashboard.close()


def test_dashboard_reloads_the_latest_checkpoint(dummy_video, tmp_path) -> None:
    loaded: list[str] = []

    def factory(path):
        loaded.append(path.name)
        return _StubPolicy()

    dashboard = TrainingDashboard(
        tmp_path,
        panels=1,
        screen_size=(640, 360),
        policy_factory=factory,
    )
    try:
        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "run_50000_steps.zip").touch()
        dashboard.reload_policy()
        assert dashboard.checkpoint_path is not None
        assert loaded == ["run_50000_steps.zip"]

        (checkpoints / "run_100000_steps.zip").touch()
        dashboard.reload_policy()
        assert loaded == ["run_50000_steps.zip", "run_100000_steps.zip"]
    finally:
        dashboard.close()


def test_dashboard_drive_mode_runs_drives(dummy_video, tmp_path) -> None:
    dashboard = TrainingDashboard(tmp_path, env_kind="drive", panels=1, screen_size=(640, 360))
    try:
        for _ in range(80):
            dashboard.step_frames(0.05)
        dashboard.draw_frame()

        panel = dashboard.panels[0]
        assert panel.down_label.startswith(("1st", "2nd", "3rd"))
        assert panel.play.state.elapsed_seconds >= 0.0
    finally:
        dashboard.close()


def test_dashboard_navigates_checkpoints(dummy_video, tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "run_50000_steps.zip").touch()
    (checkpoints / "run_100000_steps.zip").touch()
    (tmp_path / "config.json").write_text('{"total_timesteps": 200000}', encoding="utf-8")
    dashboard = TrainingDashboard(
        tmp_path,
        panels=1,
        screen_size=(640, 360),
        policy_factory=lambda path: _StubPolicy(),
    )
    try:
        dashboard.reload_policy()
        assert dashboard.checkpoint_path.name == "run_100000_steps.zip"
        assert dashboard.checkpoint_status() == "checkpoint 2/2 · step 100,000 / 200,000"

        dashboard.select_previous_checkpoint()
        assert dashboard.checkpoint_path.name == "run_50000_steps.zip"
        assert dashboard.checkpoint_status() == "checkpoint 1/2 · step 50,000 / 200,000"

        dashboard.follow_live()
        dashboard.reload_policy()
        assert dashboard.checkpoint_path.name == "run_100000_steps.zip"
    finally:
        dashboard.close()


def test_dashboard_arrow_keys_change_checkpoints(dummy_video, tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "run_50000_steps.zip").touch()
    (checkpoints / "run_100000_steps.zip").touch()
    dashboard = TrainingDashboard(
        tmp_path,
        panels=1,
        screen_size=(640, 360),
        policy_factory=lambda path: _StubPolicy(),
    )
    try:
        dashboard.reload_policy()

        dashboard._handle_events(
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)]
        )
        assert dashboard.checkpoint_path.name == "run_50000_steps.zip"

        dashboard._handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l)])
        assert dashboard.checkpoint_path.name == "run_100000_steps.zip"
    finally:
        dashboard.close()


def test_dashboard_role_play_mode_runs_plays(dummy_video, tmp_path) -> None:
    dashboard = TrainingDashboard(
        tmp_path,
        env_kind="role_play",
        learner="wr1",
        opponent_policies={"qb": _rule_qb()},
        panels=1,
        screen_size=(640, 360),
        policy_factory=lambda path: _StubRolePolicy(),
    )
    try:
        for _ in range(120):
            dashboard.step_frames(0.05)
        dashboard.draw_frame()

        panel = dashboard.panels[0]
        assert panel.plays >= 1
        assert panel.last_outcome != ""
    finally:
        dashboard.close()


def test_dashboard_role_drive_mode_runs_drives(dummy_video, tmp_path) -> None:
    dashboard = TrainingDashboard(
        tmp_path,
        env_kind="role_drive",
        learner="cb1",
        opponent_policies={"qb": _rule_qb()},
        panels=1,
        screen_size=(640, 360),
        policy_factory=lambda path: _StubRolePolicy(),
    )
    try:
        for _ in range(120):
            dashboard.step_frames(0.05)
        dashboard.draw_frame()

        panel = dashboard.panels[0]
        assert panel.drive.state.play_number >= 1
        assert panel.down_label.startswith(("1st", "2nd", "3rd"))
    finally:
        dashboard.close()
