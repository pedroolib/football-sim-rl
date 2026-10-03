import pytest

pygame = pytest.importorskip("pygame")

from qb_sim.agents.rule_controllers import HeuristicQuarterback  # noqa: E402
from qb_sim.rules.drive import DriveConfig  # noqa: E402
from qb_sim.rules.scenarios import get_scenario  # noqa: E402
from qb_sim.ui.dashboard import TrainingDashboard  # noqa: E402
from qb_sim.ui.menu import ModelMenu, ModelOption, ParameterMenu  # noqa: E402


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


def test_dashboard_mode_label(dummy_video, tmp_path) -> None:
    dashboard = TrainingDashboard(
        tmp_path,
        panels=1,
        screen_size=(320, 240),
        training=True,
    )
    try:
        assert dashboard.mode_label() == "entrenando"
    finally:
        dashboard.close()

    dashboard = TrainingDashboard(tmp_path, panels=1, screen_size=(320, 240))
    try:
        assert dashboard.mode_label() == "demo"
    finally:
        dashboard.close()


def _composition_menus():
    scenario = get_scenario("stage_b")
    menu = ModelMenu({role: [ModelOption("reglas", None)] for role in ModelMenu.ROLES})
    parameters = ParameterMenu(scenario.config, DriveConfig(los_range=(40.0, 40.0)))
    return menu, parameters


def test_dashboard_composition_mode_runs_drives(dummy_video, tmp_path) -> None:
    menu, parameters = _composition_menus()
    calls: list[int] = []

    def qb(state, routes, drive_state):
        calls.append(1)
        return None

    dashboard = TrainingDashboard(
        tmp_path,
        env_kind="composition",
        panels=1,
        screen_size=(640, 360),
        opponent_policies={"qb": qb},
        model_menu=menu,
        parameter_menu=parameters,
    )
    try:
        for _ in range(200):
            dashboard.step_frames(0.05)
        dashboard.draw_frame()

        assert calls
        assert dashboard.panels[0].drive.state.play_number >= 1
    finally:
        dashboard.close()


def test_dashboard_parameter_menu_adjusts_the_scenario(dummy_video, tmp_path) -> None:
    menu, parameters = _composition_menus()
    dashboard = TrainingDashboard(
        tmp_path,
        env_kind="composition",
        panels=1,
        screen_size=(640, 360),
        model_menu=menu,
        parameter_menu=parameters,
    )
    try:
        dashboard._handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m)])
        assert dashboard._menu_mode == "params"

        index = next(
            i for i, entry in enumerate(parameters.entries) if entry.name == "receiver_speed"
        )
        parameters.index = index
        before = parameters.sim_config().receiver_speed

        dashboard._handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)])

        assert parameters.sim_config().receiver_speed > before
        assert dashboard.scenario.config.receiver_speed > before
    finally:
        dashboard.close()


def test_dashboard_model_menu_cycles_the_focused_role(dummy_video, tmp_path) -> None:
    menu, parameters = _composition_menus()
    dashboard = TrainingDashboard(
        tmp_path,
        env_kind="composition",
        panels=1,
        screen_size=(640, 360),
        model_menu=menu,
        parameter_menu=parameters,
    )
    try:
        dashboard._handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n)])
        assert dashboard._menu_mode == "models"

        dashboard._handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)])

        assert menu.current("qb").label == "reglas"
    finally:
        dashboard.close()


def test_dashboard_composition_ignores_checkpoint_keys(dummy_video, tmp_path) -> None:
    menu, parameters = _composition_menus()
    dashboard = TrainingDashboard(
        tmp_path,
        env_kind="composition",
        panels=1,
        screen_size=(640, 360),
        model_menu=menu,
        parameter_menu=parameters,
    )
    try:
        dashboard._handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)])
        dashboard._handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)])

        assert dashboard.checkpoint_path is None
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
