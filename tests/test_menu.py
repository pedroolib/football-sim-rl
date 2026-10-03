import json
from pathlib import Path

from qb_sim.config import SimulationConfig
from qb_sim.rules.drive import DriveConfig
from qb_sim.ui.menu import ModelMenu, ParameterMenu


def test_parameter_menu_lists_numeric_config_fields() -> None:
    menu = ParameterMenu(SimulationConfig(), DriveConfig())
    names = [entry.name for entry in menu.entries]

    assert "receiver_speed" in names
    assert "rusher_speed" in names
    assert "carrier_fatigue_per_second" in names
    assert "los_min" in names
    assert "los_max" in names


def test_parameter_menu_adjusts_a_speed() -> None:
    menu = ParameterMenu(SimulationConfig(), DriveConfig())
    index = next(i for i, entry in enumerate(menu.entries) if entry.name == "receiver_speed")
    menu.index = index

    before = menu.sim_config().receiver_speed
    assert menu.adjust(1)

    assert menu.sim_config().receiver_speed > before
    assert menu.entries[index].value > before


def test_parameter_menu_keeps_the_value_when_validation_fails() -> None:
    menu = ParameterMenu(SimulationConfig(), DriveConfig())
    index = next(
        i for i, entry in enumerate(menu.entries) if entry.name == "carrier_min_speed_factor"
    )
    menu.index = index
    menu.entries[index].value = 1.0

    assert not menu.adjust(1) or menu.sim_config().carrier_min_speed_factor <= 1.0


def test_parameter_menu_moves_the_rusher_start() -> None:
    config = SimulationConfig(rusher_starts=((30.0, 29.0),))
    menu = ParameterMenu(config, DriveConfig())
    index = next(i for i, entry in enumerate(menu.entries) if entry.name == "rusher_distance")
    menu.index = index

    menu.adjust(1)

    assert menu.sim_config().rusher_starts[0][0] > 30.0


def test_parameter_menu_adjusts_the_los_range() -> None:
    menu = ParameterMenu(SimulationConfig(), DriveConfig())
    index = next(i for i, entry in enumerate(menu.entries) if entry.name == "los_min")
    menu.index = index

    menu.adjust(1)

    assert menu.drive_config().los_range[0] > 25.0


def test_model_menu_discovers_runs_by_config(tmp_path) -> None:
    def _write(run, kind, learner):
        directory = tmp_path / run
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"env_kind": kind, "learner": learner}), encoding="utf-8"
        )
        (directory / "final_model.zip").touch()

    _write("ppo_drive_qb_v12", "drive", "qb")
    _write("ppo_wr_v1", "role_play", "wr1")
    _write("ppo_cb_v1", "role_drive", "cb1")
    _write("ppo_ru_v1_bc", "role_drive", "ru1")

    menu = ModelMenu.discover(tmp_path)

    assert menu.labels("qb") == ["reglas", "ppo_drive_qb_v12"]
    assert menu.labels("wr1") == ["reglas", "ppo_wr_v1"]
    assert menu.labels("cb2") == ["reglas", "ppo_cb_v1"]
    assert menu.labels("ru1") == ["reglas", "ppo_ru_v1_bc"]


def test_model_menu_discovers_nested_position_runs(tmp_path) -> None:
    def _write(relative, kind, learner):
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(
            json.dumps({"env_kind": kind, "learner": learner}), encoding="utf-8"
        )
        (directory / "final_model.zip").touch()

    _write("qb/drive/trained_roles_v12", "drive", "qb")
    _write("wr/route_progress_v1", "role_play", "wr1")
    _write("cb/context_shaping_v1", "role_drive", "cb1")
    _write("ru/bc_parity_v1", "role_drive", "ru1")

    menu = ModelMenu.discover(tmp_path)

    assert menu.labels("qb") == ["reglas", "trained_roles_v12"]
    assert menu.labels("wr1") == ["reglas", "route_progress_v1"]
    assert menu.labels("cb2") == ["reglas", "context_shaping_v1"]
    assert menu.labels("ru1") == ["reglas", "bc_parity_v1"]


def test_model_menu_skips_history_and_pilots(tmp_path) -> None:
    def _write(relative, kind, learner):
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(
            json.dumps({"env_kind": kind, "learner": learner}), encoding="utf-8"
        )
        (directory / "final_model.zip").touch()

    _write("qb/drive/trained_roles_v12", "drive", "qb")
    _write("qb/play/history/baseline_v10", "passing", "qb")
    _write("qb/drive/pilots/random_routes_v11_pilot", "drive", "qb")

    menu = ModelMenu.discover(tmp_path)

    assert menu.labels("qb") == ["reglas", "trained_roles_v12"]


def test_model_menu_cycles_and_chooses_current_path(tmp_path) -> None:
    directory = tmp_path / "ppo_wr_v1"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"env_kind": "role_play", "learner": "wr1"}), encoding="utf-8"
    )
    (directory / "final_model.zip").touch()
    menu = ModelMenu.discover(tmp_path)

    assert menu.current("wr1").path is None
    menu.cycle("wr1", 1)
    assert menu.current("wr1").path == directory / "final_model.zip"
    menu.cycle("wr1", 1)
    assert menu.current("wr1").path is None
    assert menu.choose("wr1", "ppo_wr_v1")
    assert menu.current("wr1").path is not None


def test_model_menu_paths_reflect_choices(tmp_path) -> None:
    paths = ModelMenu.discover(tmp_path).paths()
    assert paths["qb"] is None
    assert Path("missing") not in [path for path in paths.values() if path is not None]
