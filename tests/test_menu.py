import json
import zipfile
from pathlib import Path

from qb_sim.config import SimulationConfig
from qb_sim.rules.drive import DriveConfig
from qb_sim.ui.menu import (
    ModelMenu,
    ModelOption,
    ParameterMenu,
    model_observation_size,
)


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


def _fake_model(path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data",
            json.dumps({"observation_space": {"_shape": [size]}}),
        )


def test_model_observation_size_reads_the_saved_space(tmp_path) -> None:
    model = tmp_path / "final_model.zip"
    _fake_model(model, 35)
    assert model_observation_size(model) == 35


def test_model_observation_size_returns_none_for_a_broken_zip(tmp_path) -> None:
    model = tmp_path / "final_model.zip"
    model.write_text("not a zip")
    assert model_observation_size(model) is None


def test_discover_annotates_the_observation_size(tmp_path) -> None:
    run = tmp_path / "wr" / "route_progress_v1"
    _fake_model(run / "final_model.zip", 31)
    (run / "config.json").write_text(
        json.dumps({"env_kind": "role_play", "learner": "wr1"}),
        encoding="utf-8",
    )
    menu = ModelMenu.discover(tmp_path)
    sizes = {option.label: option.observation_size for option in menu.options["wr1"]}
    assert sizes["route_progress_v1"] == 31


def test_discover_annotates_the_run_folder(tmp_path) -> None:
    nested = tmp_path / "wr" / "route_progress_v1"
    _fake_model(nested / "final_model.zip", 31)
    (nested / "config.json").write_text(
        json.dumps({"env_kind": "role_play", "learner": "wr1"}),
        encoding="utf-8",
    )
    root = tmp_path / "ppo_wr_g1"
    _fake_model(root / "final_model.zip", 35)
    (root / "config.json").write_text(
        json.dumps({"env_kind": "role_play", "learner": "wr1"}),
        encoding="utf-8",
    )
    menu = ModelMenu.discover(tmp_path)

    groups = {
        option.label: option.group
        for option in menu.options["wr1"]
    }
    assert groups["route_progress_v1"] == "wr"
    assert groups["ppo_wr_g1"] == "runs"
    assert {
        group: [option.label for option in options]
        for group, options in menu.folders("wr1")
    } == {
        "reglas": ["reglas"],
        "runs": ["ppo_wr_g1"],
        "wr": ["route_progress_v1"],
    }
    assert menu.choose("wr1", "route_progress_v1")
    assert ModelMenu.display(menu.current("wr1")) == "wr/route_progress_v1"
    assert menu.choose("wr1", "ppo_wr_g1")
    assert ModelMenu.display(menu.current("wr1")) == "runs/ppo_wr_g1"


def test_choose_compatible_prefers_a_compatible_hint() -> None:
    menu = ModelMenu(
        {
            role: [
                ModelOption("reglas", None),
                ModelOption("old", None, observation_size=31),
                ModelOption("new", None, observation_size=35),
            ]
            for role in ModelMenu.ROLES
        }
    )
    assert menu.choose_compatible("wr1", (35, 31), hints=("old",))
    assert menu.current("wr1").label == "old"
    assert not menu.choose_compatible("qb", (61,))
    assert menu.current("qb").label == "reglas"


def test_choose_compatible_prefers_the_current_size_without_hints() -> None:
    menu = ModelMenu(
        {
            role: [
                ModelOption("reglas", None),
                ModelOption("old", None, observation_size=31),
                ModelOption("new", None, observation_size=35),
            ]
            for role in ModelMenu.ROLES
        }
    )
    assert menu.choose_compatible("wr1", (35, 31))
    assert menu.current("wr1").label == "new"


def test_discover_includes_receiver_drive_runs(tmp_path) -> None:
    run = tmp_path / "wr" / "drive_v1"
    _fake_model(run / "final_model.zip", 40)
    (run / "config.json").write_text(
        json.dumps({"env_kind": "receiver_drive", "learner": "wr1"}),
        encoding="utf-8",
    )
    menu = ModelMenu.discover(tmp_path)

    assert menu.labels("wr1") == ["reglas", "drive_v1"]
    assert menu.labels("wr2") == ["reglas", "drive_v1"]
    assert menu.options["wr1"][1].observation_size == 40


def test_choose_compatible_falls_back_to_a_legacy_size() -> None:
    menu = ModelMenu(
        {
            role: [
                ModelOption("reglas", None),
                ModelOption("old", None, observation_size=31),
            ]
            for role in ModelMenu.ROLES
        }
    )
    assert menu.choose_compatible("wr1", (35, 31), hints=("new", "old"))
    assert menu.current("wr1").label == "old"


def test_choose_compatible_honors_the_hint_when_sizes_match() -> None:
    menu = ModelMenu(
        {
            role: [
                ModelOption("reglas", None),
                ModelOption("hinted", None, observation_size=35),
                ModelOption("other", None, observation_size=35),
            ]
            for role in ModelMenu.ROLES
        }
    )
    assert menu.choose_compatible("wr1", (35,), hints=("other",))
    assert menu.current("wr1").label == "other"


def test_model_menu_tree_starts_collapsed_and_toggles() -> None:
    menu = ModelMenu(
        {
            role: [ModelOption("reglas", None, group="reglas")]
            for role in ModelMenu.ROLES
        }
    )
    assert [row[0] for row in menu.visible_rows()] == ["role"] * len(ModelMenu.ROLES)
    assert menu.role() == "qb"

    assert menu.toggle() == "expand"
    kinds = [row[0] for row in menu.visible_rows()]
    assert kinds[:3] == ["role", "folder", "role"]

    menu.move(1)
    assert menu.focused_row()[0] == "folder"
    assert menu.toggle() == "expand"
    assert menu.visible_rows()[2][0] == "option"

    menu.move(1)
    assert menu.toggle() == "select"
    assert menu.current("qb").label == "reglas"
    menu.move(-1)
    assert menu.focused_row()[0] == "folder"
    assert menu.toggle() == "expand"
    assert [row[0] for row in menu.visible_rows()][:3] == ["role", "folder", "role"]


def test_model_menu_move_clamps_the_cursor() -> None:
    menu = ModelMenu(
        {
            role: [ModelOption("reglas", None, group="reglas")]
            for role in ModelMenu.ROLES
        }
    )
    menu.move(-5)
    assert menu.cursor == 0
    menu.move(100)
    assert menu.cursor == len(ModelMenu.ROLES) - 1
    assert menu.role() == ModelMenu.ROLES[-1]


def test_model_menu_focus_current_reveals_the_selected_model() -> None:
    menu = ModelMenu(
        {
            role: [
                ModelOption("reglas", None, group="reglas"),
                ModelOption("deep", None, observation_size=35, group="wr"),
            ]
            for role in ModelMenu.ROLES
        }
    )
    assert menu.choose("wr2", "deep")
    menu.cursor = ModelMenu.ROLES.index("wr2")

    menu.focus_current()

    kind, role, group, option = menu.focused_row()
    assert (kind, role, group) == ("option", "wr2", "wr")
    assert option == menu.current("wr2")
