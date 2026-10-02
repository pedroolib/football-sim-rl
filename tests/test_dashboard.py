import pytest

pygame = pytest.importorskip("pygame")

from qb_sim.ui.dashboard import TrainingDashboard  # noqa: E402


class _StubPolicy:
    def __call__(self, state):
        return None


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
