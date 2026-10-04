"""Pure menu logic for live model selection and simulation parameters."""

import json
import zipfile
from dataclasses import dataclass, fields, replace
from pathlib import Path

from qb_sim.config import SimulationConfig
from qb_sim.rules.drive import DriveConfig

_DRIVE_SCALARS = ("downs", "first_down_yards", "max_plays", "touchdown_points", "goal_line")


def _step_for(name: str, value: float | int) -> float:
    if name == "dt":
        return 0.005
    if name.endswith("_degrees"):
        return 5.0
    if name.endswith("_ratio"):
        return 0.05
    if name.endswith("_seconds"):
        return 0.1
    if name.endswith("_sigma"):
        return 0.05
    if "speed" in name:
        return 0.1
    if "yards" in name or "radius" in name or "height" in name:
        return 0.5
    if isinstance(value, int):
        return 1.0
    return 0.05


def _bounds_for(name: str, value: float | int) -> tuple[float | None, float | None]:
    if name == "dt":
        return (0.005, 0.2)
    if name.endswith("_degrees"):
        return (0.0, 90.0)
    if name.endswith("_ratio"):
        return (0.0, 1.0)
    if "speed" in name:
        return (0.5, 12.0)
    if "seconds" in name:
        return (0.1, 30.0)
    if "radius" in name or "height" in name or "yards" in name:
        return (0.0, 60.0)
    if name.startswith("carrier_") or name.endswith("_sigma"):
        return (0.0, 1.0)
    return (None, None)


@dataclass(slots=True)
class ParameterEntry:
    """One editable simulation parameter."""

    name: str
    value: float | int
    step: float
    minimum: float | None = None
    maximum: float | None = None


class ParameterMenu:
    """Editable view over a ``SimulationConfig`` and a ``DriveConfig``."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        drive_config: DriveConfig | None = None,
    ) -> None:
        self._config = config if config is not None else SimulationConfig()
        self._drive = drive_config if drive_config is not None else DriveConfig()
        self.index = 0
        self.entries = self._build_entries()

    def _build_entries(self) -> list[ParameterEntry]:
        entries: list[ParameterEntry] = []
        for item in fields(self._config):
            value = getattr(self._config, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            minimum, maximum = _bounds_for(item.name, value)
            entries.append(
                ParameterEntry(
                    item.name,
                    value,
                    _step_for(item.name, value),
                    minimum,
                    maximum,
                )
            )
        for name in _DRIVE_SCALARS:
            value = getattr(self._drive, name)
            entries.append(ParameterEntry(name, value, _step_for(name, value)))
        low, high = self._drive.los_range
        entries.append(ParameterEntry("los_min", low, 1.0, 1.0, high))
        entries.append(ParameterEntry("los_max", high, 1.0, low, self._drive.goal_line - 1.0))
        if self._config.rusher_starts:
            rusher = self._config.rusher_starts[0]
            entries.append(
                ParameterEntry(
                    "rusher_distance",
                    rusher[0] - self._config.quarterback_start[0],
                    0.5,
                    -10.0,
                    60.0,
                )
            )
            entries.append(
                ParameterEntry(
                    "rusher_lateral",
                    rusher[1],
                    0.5,
                    0.0,
                    self._config.field.width_yards,
                )
            )
        return entries

    def move(self, direction: int) -> None:
        """Move the selection by ``direction`` rows, clamped to the list."""
        if not self.entries:
            return
        self.index = max(0, min(self.index + direction, len(self.entries) - 1))

    def adjust(self, direction: int) -> bool:
        """Adjust the selected parameter; keep the old value when invalid."""
        entry = self.entries[self.index]
        new_value = entry.value + direction * entry.step
        if entry.minimum is not None:
            new_value = max(new_value, entry.minimum)
        if entry.maximum is not None:
            new_value = min(new_value, entry.maximum)
        if self._apply(entry.name, new_value):
            entry.value = new_value
            return True
        return False

    def _apply(self, name: str, value: float) -> bool:
        sim_fields = {item.name for item in fields(self._config)}
        try:
            if name in sim_fields:
                current = getattr(self._config, name)
                if isinstance(current, int) and not isinstance(current, bool):
                    value = int(round(value))
                self._config = replace(self._config, **{name: value})
                return True
            if name in _DRIVE_SCALARS:
                current = getattr(self._drive, name)
                if isinstance(current, int) and not isinstance(current, bool):
                    value = int(round(value))
                self._drive = replace(self._drive, **{name: value})
                return True
            if name in ("los_min", "los_max"):
                low, high = self._drive.los_range
                new_low, new_high = (value, high) if name == "los_min" else (low, value)
                if new_low > new_high:
                    return False
                self._drive = replace(self._drive, los_range=(new_low, new_high))
                return True
            if name == "rusher_distance":
                x, y = self._config.rusher_starts[0]
                new_x = self._config.quarterback_start[0] + value
                self._config = replace(self._config, rusher_starts=((new_x, y),))
                return True
            if name == "rusher_lateral":
                x, y = self._config.rusher_starts[0]
                self._config = replace(self._config, rusher_starts=((x, value),))
                return True
        except ValueError:
            return False
        return False

    def sim_config(self) -> SimulationConfig:
        """Current simulation configuration."""
        return self._config

    def drive_config(self) -> DriveConfig:
        """Current drive configuration."""
        return self._drive

    def label(self, index: int | None = None) -> str:
        """Human-readable line for one entry."""
        entry = self.entries[self.index if index is None else index]
        if isinstance(entry.value, int):
            return f"{entry.name}: {entry.value}"
        return f"{entry.name}: {entry.value:.2f}"


@dataclass(frozen=True, slots=True)
class ModelOption:
    """One selectable policy for a role."""

    label: str
    path: Path | None
    observation_size: int | None = None
    group: str = ""


def model_observation_size(path: str | Path) -> int | None:
    """Read the flattened observation size stored inside a saved SB3 model."""
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
        shape = data["observation_space"]["_shape"]
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
        return None
    if not shape:
        return None
    size = 1
    for dimension in shape:
        size *= int(dimension)
    return size


class ModelMenu:
    """Selectable trained models per player role."""

    ROLES = ("qb", "wr1", "wr2", "cb1", "cb2", "ru1")

    def __init__(self, options: dict[str, list[ModelOption]]) -> None:
        self.options = options
        self.selection = {role: 0 for role in options}
        self.index = 0

    @classmethod
    def discover(cls, runs_dir: str | Path = "runs") -> "ModelMenu":
        """Build the role options from the run directories on disk.

        Runs live under ``runs/<position>/...``; ``history`` and ``pilots``
        folders are skipped so the menu only offers production models.
        """
        options = {
            role: [ModelOption("reglas", None, group="reglas")] for role in cls.ROLES
        }
        root = Path(runs_dir)
        if not root.is_dir():
            return cls(options)
        for metadata_path in sorted(root.rglob("config.json")):
            run = metadata_path.parent
            if any(part in ("history", "pilots", "reports") for part in run.parts):
                continue
            model_path = run / "final_model.zip"
            if not model_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            kind = metadata.get("env_kind")
            learner = metadata.get("learner")
            size = model_observation_size(model_path)
            relative = run.parent.relative_to(root)
            group = "runs" if relative == Path(".") else relative.as_posix()
            if kind == "drive" and learner == "qb":
                options["qb"].append(ModelOption(run.name, model_path, size, group))
            elif kind in ("role_play", "receiver_drive") and learner in ("wr1", "wr2"):
                options["wr1"].append(ModelOption(run.name, model_path, size, group))
                options["wr2"].append(ModelOption(run.name, model_path, size, group))
            elif kind == "role_drive" and learner in ("cb1", "cb2"):
                options["cb1"].append(ModelOption(run.name, model_path, size, group))
                options["cb2"].append(ModelOption(run.name, model_path, size, group))
            elif kind == "role_drive" and learner == "ru1":
                options["ru1"].append(ModelOption(run.name, model_path, size, group))
        return cls(options)

    def role(self) -> str:
        """Currently focused role."""
        return list(self.options)[self.index]

    def move(self, direction: int) -> None:
        """Move the role selection."""
        self.index = max(0, min(self.index + direction, len(self.options) - 1))

    def labels(self, role: str) -> list[str]:
        """Available labels for a role."""
        return [option.label for option in self.options[role]]

    def current(self, role: str) -> ModelOption:
        """Currently selected option for a role."""
        return self.options[role][self.selection[role]]

    def cycle(self, role: str, direction: int) -> None:
        """Cycle a role's model selection."""
        count = len(self.options[role])
        self.selection[role] = (self.selection[role] + direction) % count

    def choose(self, role: str, label: str) -> bool:
        """Select a model by label; return whether it exists."""
        for index, option in enumerate(self.options[role]):
            if option.label == label:
                self.selection[role] = index
                return True
        return False

    def choose_compatible(
        self,
        role: str,
        expected_sizes: tuple[int, ...],
        hints: tuple[str, ...] = (),
    ) -> bool:
        """Select a model whose saved observation matches a supported size.

        Hinted labels win when compatible (they are curated production names);
        otherwise the first option matching the first supported size is used,
        falling back to legacy sizes in order. Returns whether a model was
        selected.
        """
        for label in hints:
            for index, option in enumerate(self.options[role]):
                if option.label == label and option.observation_size in expected_sizes:
                    self.selection[role] = index
                    return True
        for size in expected_sizes:
            for index, option in enumerate(self.options[role]):
                if option.observation_size == size:
                    self.selection[role] = index
                    return True
        return False

    @staticmethod
    def display(option: ModelOption) -> str:
        """Label qualified by its run folder, for the model selector."""
        if not option.group or option.group == "reglas":
            return option.label
        return f"{option.group}/{option.label}"

    def folders(self, role: str) -> list[tuple[str, list[ModelOption]]]:
        """Options of one role grouped by their run folder, in discovery order."""
        groups: dict[str, list[ModelOption]] = {}
        for option in self.options[role]:
            groups.setdefault(option.group or "reglas", []).append(option)
        return list(groups.items())

    def paths(self) -> dict[str, Path | None]:
        """Current path per role (``None`` means rule controllers)."""
        return {
            role: self.current(role).path
            for role in self.options
        }
