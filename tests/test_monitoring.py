from qb_sim.rules.play import Play
from qb_sim.training.monitoring import (
    DemoPanel,
    checkpoint_steps,
    latest_checkpoint,
    read_progress,
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
