"""Live Pygame dashboard: watch a policy improve while training runs."""

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pygame

from qb_sim.rules.drive import DriveConfig
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.monitoring import (
    CompositionDemoPanel,
    DefenderPolicyCallable,
    DemoPanel,
    DriveDemoPanel,
    DriveQuarterbackPolicy,
    QuarterbackPolicy,
    ReceiverPolicyCallable,
    RoleDriveDemoPanel,
    RolePlayDemoPanel,
    checkpoint_catalog,
    checkpoint_label,
    latest_checkpoint,
    read_progress,
    rule_drive_quarterback_policy,
    rule_quarterback_policy,
    total_timesteps,
)
from qb_sim.training.policy import (
    SUPPORTED_DEFENDER_SIZES,
    SUPPORTED_DRIVE_SIZES,
    SUPPORTED_RECEIVER_SIZES,
)
from qb_sim.ui import theme
from qb_sim.ui.menu import ModelMenu, ModelOption, ParameterMenu
from qb_sim.ui.rendering import (
    draw_ball,
    draw_field,
    draw_players,
    draw_routes,
    draw_scrimmage,
)
from qb_sim.ui.view import ViewTransform

DEFAULT_SCREEN_SIZE = (1600, 900)
FPS = 60
METRICS_WIDTH = 360
GRID_COLUMNS = 3
DEFAULT_RELOAD_SECONDS = 5.0
_SPEED_KEYS = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 4}
_PREVIOUS_KEYS = (pygame.K_LEFT, pygame.K_LEFTBRACKET)
_NEXT_KEYS = (pygame.K_RIGHT, pygame.K_RIGHTBRACKET)
_MENU_KEYS = (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT)
MENU_REPEAT_DELAY = 0.35
MENU_REPEAT_INTERVAL = 0.06
MENU_ACCENT = theme.PRIMARY_SOFT
MENU_TEXT = theme.TEXT
MENU_MUTED = theme.TEXT_MUTED
MENU_SELECTED = (30, 58, 100)
MENU_CARD = theme.SURFACE
MENU_ROW = theme.SURFACE_ALT
HEADER_HEIGHT = 46
CAMERA_FOLLOW_RATE = 4.0
ENV_KINDS = ("passing", "drive", "role_play", "role_drive", "composition")
LearnerPolicy = (
    QuarterbackPolicy
    | DriveQuarterbackPolicy
    | ReceiverPolicyCallable
    | DefenderPolicyCallable
)


class TrainingDashboard:
    """Grid of parallel demo plays fed by the latest training checkpoint."""

    def __init__(
        self,
        run_dir: str | Path,
        scenario_name: str = "rushed",
        panels: int = 6,
        screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
        reload_seconds: float = DEFAULT_RELOAD_SECONDS,
        policy_factory: Callable[[Path], LearnerPolicy] | None = None,
        env_kind: str = "passing",
        learner: str = "wr1",
        opponent_policies: dict | None = None,
        training: bool = False,
        qb_model: str | Path | None = None,
        model_menu: ModelMenu | None = None,
        parameter_menu: ParameterMenu | None = None,
    ) -> None:
        if env_kind not in ENV_KINDS:
            raise ValueError(f"unknown env_kind: {env_kind!r}")
        pygame.init()
        self.run_dir = Path(run_dir)
        self.env_kind = env_kind
        self.learner = learner
        self.training = training
        self.scenario = get_scenario(scenario_name)
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption(f"QB Lab — {self.run_dir.name}")
        self.clock = pygame.time.Clock()
        self.font_sans, self.font_mono, self.font_small = theme.load_fonts()
        self.font = self.font_sans
        self.reload_seconds = reload_seconds
        self._policy_factory = policy_factory or self._default_policy_factory
        self.running = True
        self.paused = False
        self.speed = 1
        self._accumulator = 0.0
        self._last_reload = 0.0
        self._checkpoint_path: Path | None = None
        self._catalog: list[Path] = checkpoint_catalog(self.run_dir)
        self._live = True
        self._opponents = dict(opponent_policies or {})
        self._menu_mode: str | None = None
        self._repeat_timers: dict[int, float] = {}
        self._parameter_menu: ParameterMenu | None = parameter_menu
        self._model_menu: ModelMenu | None = model_menu
        self._composition_policies: dict = {}
        if env_kind == "drive":
            self._policy: LearnerPolicy | None = rule_drive_quarterback_policy(
                self.scenario.config
            )
            self._panels: list[
                DemoPanel
                | DriveDemoPanel
                | RolePlayDemoPanel
                | RoleDriveDemoPanel
                | CompositionDemoPanel
            ] = [
                DriveDemoPanel(self.scenario, policy=self._policy, seed=1000 + index)
                for index in range(panels)
            ]
        elif env_kind == "role_play":
            self._policy = None
            self._panels = [
                RolePlayDemoPanel(
                    self.scenario,
                    learner_id=self.learner,
                    policy=None,
                    opponents=self._opponents,
                    seed=1000 + index,
                )
                for index in range(panels)
            ]
        elif env_kind == "role_drive":
            self._policy = None
            self._panels = [
                RoleDriveDemoPanel(
                    self.scenario,
                    learner_id=self.learner,
                    policy=None,
                    opponents=self._opponents,
                    seed=1000 + index,
                )
                for index in range(panels)
            ]
        elif env_kind == "composition":
            self._parameter_menu = self._parameter_menu or ParameterMenu(
                self.scenario.config, DriveConfig()
            )
            self._model_menu = self._model_menu or ModelMenu.discover()
            self._prefer_default_models(qb_model)
            self._composition_policies = self._build_composition_policies()
            qb_policy = self._composition_policies.get("qb")
            opponents = {
                role: policy
                for role, policy in self._composition_policies.items()
                if role != "qb"
            }
            self._policy = qb_policy
            self._panels = [
                CompositionDemoPanel(
                    self.scenario,
                    qb_policy=qb_policy,
                    opponents=opponents,
                    seed=1000 + index,
                    drive_config=self._parameter_menu.drive_config(),
                )
                for index in range(panels)
            ]
        else:
            self._policy = rule_quarterback_policy(self.scenario.config)
            self._panels = [
                DemoPanel(
                    Play(self.scenario.config, seed=index),
                    self.scenario.receiver_routes,
                    policy=self._policy,
                    seed=1000 + index,
                )
                for index in range(panels)
            ]
        self._grid = self._compute_grid(panels, screen_size, header_height=HEADER_HEIGHT)
        field = self.scenario.config.field
        self._cameras: list[tuple[float, float]] = [
            (field.length_yards / 2.0, field.width_yards / 2.0) for _ in range(panels)
        ]

    @property
    def panels(
        self,
    ) -> list[
        DemoPanel
        | DriveDemoPanel
        | RolePlayDemoPanel
        | RoleDriveDemoPanel
        | CompositionDemoPanel
    ]:
        """Demo panels in display order."""
        return self._panels

    def _prefer_default_models(self, qb_model: str | Path | None) -> None:
        if self._model_menu is None:
            return
        if qb_model is not None:
            self._model_menu.choose("qb", Path(qb_model).parent.name)
        else:
            self._model_menu.choose_compatible(
                "qb",
                SUPPORTED_DRIVE_SIZES,
                hints=("adaptation_g1", "ppo_qb_g1", "trained_roles_v12"),
            )
        defaults = {
            "wr1": (
                SUPPORTED_RECEIVER_SIZES,
                ("adaptation_g1", "ppo_wr_g1", "route_progress_v1"),
            ),
            "wr2": (
                SUPPORTED_RECEIVER_SIZES,
                ("adaptation_g1", "ppo_wr_g1", "route_progress_v1"),
            ),
            "cb1": (
                SUPPORTED_DEFENDER_SIZES,
                ("adaptation_g1", "ppo_cb_g1", "context_shaping_v1"),
            ),
            "cb2": (
                SUPPORTED_DEFENDER_SIZES,
                ("adaptation_g1", "ppo_cb_g1", "context_shaping_v1"),
            ),
            "ru1": (
                SUPPORTED_DEFENDER_SIZES,
                ("adaptation_g1", "ppo_ru_g1", "bc_parity_v1", "lr5_interception_v1"),
            ),
        }
        for role, (sizes, labels) in defaults.items():
            self._model_menu.choose_compatible(role, sizes, hints=labels)

    def _build_composition_policies(self) -> dict:
        if self._model_menu is None:
            return {}
        from qb_sim.training.policy import (
            load_defender_policy,
            load_drive_policy,
            load_receiver_policy,
        )

        policies: dict = {}
        for role, path in self._model_menu.paths().items():
            if path is None:
                continue
            try:
                if role == "qb":
                    policies[role] = load_drive_policy(path, self.scenario.name)
                elif role in ("wr1", "wr2"):
                    policies[role] = load_receiver_policy(path, self.scenario.name, role)
                else:
                    policies[role] = load_defender_policy(path, self.scenario.name, role)
            except Exception as error:  # noqa: BLE001 - model may be mid-write
                print(f"modelo no cargado ({role}, {path.name}): {error}")
        policies.update(self._opponents)
        return policies

    def _rebuild_composition(self) -> None:
        if self._parameter_menu is None:
            return
        self.scenario = replace(self.scenario, config=self._parameter_menu.sim_config())
        drive_config = self._parameter_menu.drive_config()
        qb_policy = self._composition_policies.get("qb")
        opponents = {
            role: policy
            for role, policy in self._composition_policies.items()
            if role != "qb"
        }
        self._policy = qb_policy
        for panel in self._panels:
            if isinstance(panel, CompositionDemoPanel):
                panel.rebuild(self.scenario, drive_config, qb_policy, opponents)

    def _reload_composition(self) -> None:
        self._composition_policies = self._build_composition_policies()
        self._rebuild_composition()

    @property
    def checkpoint_path(self) -> Path | None:
        """Currently loaded checkpoint, if any."""
        return self._checkpoint_path

    @property
    def policy(self) -> LearnerPolicy | None:
        """Learner policy currently used by the panels."""
        return self._policy

    @property
    def live(self) -> bool:
        """Whether the dashboard follows the newest checkpoint."""
        return self._live

    def mode_label(self) -> str:
        """Whether the dashboard is training or replaying saved policies."""
        if self.env_kind == "composition":
            return "composición"
        return "entrenando" if self.training else "demo"

    def _default_policy_factory(self, path: Path) -> LearnerPolicy:
        if self.env_kind == "role_play":
            from qb_sim.training.policy import load_receiver_policy

            return load_receiver_policy(path, self.scenario.name, self.learner)
        if self.env_kind == "role_drive":
            from qb_sim.training.policy import load_defender_policy

            return load_defender_policy(path, self.scenario.name, self.learner)
        if self.env_kind in ("drive", "composition"):
            from qb_sim.training.policy import load_drive_policy

            return load_drive_policy(path, self.scenario.name)
        from qb_sim.training.policy import load_policy

        return load_policy(path, self.scenario.name)

    def _load_checkpoint(self, path: Path) -> None:
        try:
            policy = self._policy_factory(path)
        except Exception as error:  # noqa: BLE001 - checkpoint may be mid-write
            print(f"checkpoint no cargado ({path.name}): {error}")
            return
        self._policy = policy
        self._checkpoint_path = path
        for panel in self._panels:
            panel.set_policy(policy)

    def _catalog_index(self) -> int | None:
        if self._checkpoint_path is None or self._checkpoint_path not in self._catalog:
            return None
        return self._catalog.index(self._checkpoint_path)

    def _refresh_catalog(self) -> None:
        self._catalog = checkpoint_catalog(self.run_dir)

    def select_previous_checkpoint(self) -> None:
        """Load the previous generation in the catalog."""
        if self.env_kind == "composition":
            return
        self._refresh_catalog()
        if not self._catalog:
            return
        current = self._catalog_index()
        index = (current if current is not None else len(self._catalog)) - 1
        self._live = False
        self._load_checkpoint(self._catalog[max(index, 0)])

    def select_next_checkpoint(self) -> None:
        """Load the next generation in the catalog."""
        if self.env_kind == "composition":
            return
        self._refresh_catalog()
        if not self._catalog:
            return
        current = self._catalog_index()
        index = (current if current is not None else -1) + 1
        self._live = False
        self._load_checkpoint(self._catalog[min(index, len(self._catalog) - 1)])

    def follow_live(self) -> None:
        """Follow the newest checkpoint again."""
        self._live = True
        self.reload_policy()

    def checkpoint_status(self) -> str:
        """Human-readable generation and progress label."""
        if self.env_kind == "composition" and self._model_menu is not None:
            return f"modelos: qb={self._model_menu.current('qb').label}"
        self._refresh_catalog()
        if not self._catalog:
            return "checkpoint: reglas"
        index = self._catalog_index()
        if index is None:
            return "checkpoint: vivo"
        label = checkpoint_label(self._catalog[index])
        total = total_timesteps(self.run_dir)
        progress = f"{label} / {total:,}" if total is not None else label
        return f"checkpoint {index + 1}/{len(self._catalog)} · {progress}"

    def reload_policy(self) -> None:
        """Load the newest checkpoint when it changes (only while live)."""
        if self.env_kind == "composition":
            return
        self._refresh_catalog()
        if not self._live:
            return
        checkpoint = latest_checkpoint(self.run_dir)
        if checkpoint is None or checkpoint == self._checkpoint_path:
            return
        self._load_checkpoint(checkpoint)

    def step_frames(self, real_seconds: float) -> None:
        """Advance the demo panels at a fixed timestep, scaled by speed."""
        self._update_menu_repeat(real_seconds)
        if self.paused:
            return
        self._update_cameras(real_seconds)
        self._accumulator += real_seconds * self.speed
        dt = self.scenario.config.dt
        while self._accumulator >= dt:
            for panel in self._panels:
                panel.tick()
            self._accumulator -= dt

    def _update_cameras(self, real_seconds: float) -> None:
        factor = min(1.0, real_seconds * CAMERA_FOLLOW_RATE)
        for index, panel in enumerate(self._panels):
            ball = panel.play.state.ball.position
            camera_x, camera_y = self._cameras[index]
            self._cameras[index] = (
                camera_x + (float(ball[0]) - camera_x) * factor,
                camera_y + (float(ball[1]) - camera_y) * factor,
            )

    def _panel_view(
        self,
        rect: pygame.Rect,
        field_height: int,
        index: int,
        margin: int = 8,
    ) -> ViewTransform:
        field = self.scenario.config.field
        usable_width = rect.width - 2 * margin
        usable_height = field_height - 2 * margin
        scale = usable_height / field.width_yards
        visible_length = usable_width / scale
        if visible_length >= field.length_yards:
            return ViewTransform(field, rect.width, field_height, margin=margin)
        half_length = visible_length / 2.0
        camera_x, _ = self._cameras[index]
        camera_x = min(max(camera_x, half_length), field.length_yards - half_length)
        return ViewTransform(
            field,
            rect.width,
            field_height,
            margin=margin,
            center=(camera_x, field.width_yards / 2.0),
            visible_width_yards=field.width_yards,
        )

    def _handle_events(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYUP:
                self._repeat_timers.pop(event.key, None)
            elif event.type == pygame.KEYDOWN:
                if self._menu_mode is not None:
                    if event.key == pygame.K_ESCAPE:
                        self._menu_mode = None
                        self._repeat_timers.clear()
                    elif event.key == pygame.K_m and self.env_kind == "composition":
                        self._menu_mode = "params"
                        self._repeat_timers.clear()
                    elif event.key == pygame.K_n and self.env_kind == "composition":
                        self._menu_mode = "models"
                        self._repeat_timers.clear()
                    elif event.key in _MENU_KEYS:
                        self._apply_menu_key(event.key)
                        self._repeat_timers[event.key] = -MENU_REPEAT_DELAY
                    continue
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_m and self.env_kind == "composition":
                    self._menu_mode = "params"
                elif event.key == pygame.K_n and self.env_kind == "composition":
                    self._menu_mode = "models"
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key in _SPEED_KEYS:
                    self.speed = _SPEED_KEYS[event.key]
                elif event.key in _PREVIOUS_KEYS:
                    self.select_previous_checkpoint()
                elif event.key in _NEXT_KEYS:
                    self.select_next_checkpoint()
                elif event.key == pygame.K_l:
                    self.follow_live()
                elif event.key == pygame.K_r:
                    for index, panel in enumerate(self._panels):
                        panel.restart(seed=1000 + index)

    def _apply_menu_key(self, key: int) -> None:
        if key == pygame.K_UP:
            self._move_menu_selection(-1)
        elif key == pygame.K_DOWN:
            self._move_menu_selection(1)
        elif key == pygame.K_LEFT:
            self._adjust_menu(-1)
        elif key == pygame.K_RIGHT:
            self._adjust_menu(1)

    def _update_menu_repeat(self, real_seconds: float) -> None:
        if self._menu_mode is None:
            self._repeat_timers.clear()
            return
        for key, timer in list(self._repeat_timers.items()):
            timer += real_seconds
            while timer >= 0.0:
                self._apply_menu_key(key)
                timer -= MENU_REPEAT_INTERVAL
            self._repeat_timers[key] = timer

    def _move_menu_selection(self, direction: int) -> None:
        if self._menu_mode == "params" and self._parameter_menu is not None:
            self._parameter_menu.move(direction)
        elif self._menu_mode == "models" and self._model_menu is not None:
            self._model_menu.move(direction)

    def _adjust_menu(self, direction: int) -> None:
        if self._menu_mode == "params" and self._parameter_menu is not None:
            if self._parameter_menu.adjust(direction):
                self._rebuild_composition()
        elif self._menu_mode == "models" and self._model_menu is not None:
            self._model_menu.cycle(self._model_menu.role(), direction)
            self._reload_composition()

    def _draw_menu(self) -> None:
        rows: list[tuple[str, str, float | None, tuple[float, float] | None]] = []
        if self._menu_mode == "params" and self._parameter_menu is not None:
            title = "PARÁMETROS"
            for entry in self._parameter_menu.entries:
                value_text = (
                    str(entry.value)
                    if isinstance(entry.value, int)
                    else f"{entry.value:.2f}"
                )
                bounds = (
                    (float(entry.minimum), float(entry.maximum))
                    if entry.minimum is not None and entry.maximum is not None
                    else None
                )
                rows.append((entry.name, value_text, float(entry.value), bounds))
            selected = self._parameter_menu.index
        elif self._model_menu is not None:
            title = "MODELOS"
            for role in self._model_menu.options:
                label = ModelMenu.display(self._model_menu.current(role))
                rows.append((role, label if len(label) <= 20 else label[:18] + "…", None, None))
            selected = self._model_menu.index
        else:
            return

        model_rows = title == "MODELOS" and self._model_menu is not None

        dim = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        dim.fill((4, 8, 12, 150))
        self.screen.blit(dim, (0, 0))

        row_height = 30
        visible_rows = 14
        start = max(0, min(selected - visible_rows // 2, max(0, len(rows) - visible_rows)))
        window = rows[start : start + visible_rows]
        card_width = 780
        card_height = (len(window) + 2) * row_height + 30
        card = pygame.Rect(
            (self.screen.get_width() - card_width) // 2,
            (self.screen.get_height() - card_height) // 2,
            card_width,
            card_height,
        )
        shadow = pygame.Surface((card.width, card.height), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 110))
        self.screen.blit(shadow, (card.x + 6, card.y + 8))
        pygame.draw.rect(self.screen, MENU_CARD, card, border_radius=18)
        pygame.draw.rect(self.screen, MENU_ACCENT, card, 2, border_radius=18)

        title_image = self.font.render(title, True, MENU_TEXT)
        self.screen.blit(title_image, (card.x + 24, card.y + 16))
        hint = "Arriba/Abajo mover · Izq/Der ajustar (mantener) · M/N cambiar · Esc cerrar"
        hint_image = self.font.render(hint, True, MENU_MUTED)
        self.screen.blit(
            hint_image,
            (card.right - hint_image.get_width() - 24, card.y + 18),
        )
        pygame.draw.line(
            self.screen,
            (40, 52, 68),
            (card.x + 20, card.y + 46),
            (card.right - 20, card.y + 46),
            1,
        )

        for row, (name, value_text, numeric, bounds) in enumerate(window):
            index = start + row
            row_width = 330 if model_rows else card.width - 28
            row_rect = pygame.Rect(
                card.x + 14,
                card.y + 56 + row * row_height,
                row_width,
                row_height - 4,
            )
            if index == selected:
                pygame.draw.rect(self.screen, MENU_SELECTED, row_rect, border_radius=9)
            elif row % 2 == 1:
                pygame.draw.rect(self.screen, MENU_ROW, row_rect, border_radius=9)
            name_image = self.font.render(
                name,
                True,
                MENU_TEXT if index == selected else MENU_MUTED,
            )
            self.screen.blit(name_image, (row_rect.x + 16, row_rect.y + 6))
            value_color = MENU_ACCENT if index == selected else MENU_TEXT
            value_image = self.font_mono.render(value_text, True, value_color)
            if bounds is not None and bounds[1] > bounds[0] and numeric is not None:
                track = pygame.Rect(row_rect.x + 360, row_rect.centery - 3, 210, 6)
                pygame.draw.rect(self.screen, (44, 54, 68), track, border_radius=3)
                fraction = (numeric - bounds[0]) / (bounds[1] - bounds[0])
                fraction = max(0.0, min(1.0, fraction))
                fill = pygame.Rect(
                    track.x,
                    track.y,
                    int(track.width * fraction),
                    track.height,
                )
                pygame.draw.rect(self.screen, MENU_ACCENT, fill, border_radius=3)
                pygame.draw.circle(
                    self.screen,
                    MENU_TEXT,
                    (track.x + int(track.width * fraction), track.centery),
                    6,
                )
                value_x = track.right + 22
            else:
                value_x = row_rect.right - value_image.get_width() - 22
            self.screen.blit(value_image, (value_x, row_rect.y + 6))

        if model_rows:
            self._draw_model_folders(card)

        if len(rows) > visible_rows:
            track = pygame.Rect(card.right - 12, card.y + 56, 4, len(window) * row_height)
            pygame.draw.rect(self.screen, (44, 54, 68), track, border_radius=2)
            thumb_height = max(24, int(track.height * visible_rows / len(rows)))
            span = max(1, len(rows) - visible_rows)
            thumb_y = track.y + int((track.height - thumb_height) * start / span)
            pygame.draw.rect(
                self.screen,
                MENU_ACCENT,
                pygame.Rect(track.x, thumb_y, track.width, thumb_height),
                border_radius=2,
            )

    @staticmethod
    def _scroll_start(total: int, current: int, capacity: int) -> int:
        """First visible line so that ``current`` stays inside the window."""
        if total <= capacity:
            return 0
        start = current - capacity // 2
        return max(0, min(start, total - capacity))

    def _draw_model_folders(self, card: pygame.Rect) -> None:
        """Folder-grouped list of the focused role's models, with observation sizes.

        The list scrolls to keep the selected model visible, so every
        discovered model can be reached with the arrow keys.
        """
        if self._model_menu is None:
            return
        role = self._model_menu.role()
        current = self._model_menu.current(role)
        panel = pygame.Rect(card.x + 360, card.y + 52, card.width - 384, card.height - 64)
        pygame.draw.rect(self.screen, MENU_ROW, panel, border_radius=10)
        pygame.draw.rect(self.screen, (40, 52, 68), panel, 1, border_radius=10)
        lines: list[tuple[str, str, ModelOption | None]] = []
        for group, options in self._model_menu.folders(role):
            lines.append(("header", group, None))
            for option in options:
                lines.append(("option", group, option))
        line_height = 21
        strip = 22
        capacity = max(1, (panel.height - strip - 8) // line_height)
        current_line = next(
            (
                index
                for index, (kind, _, option) in enumerate(lines)
                if kind == "option" and option == current
            ),
            0,
        )
        start = self._scroll_start(len(lines), current_line, capacity)
        window = lines[start : start + capacity]
        y = panel.y + strip
        for kind, group, option in window:
            if kind == "header":
                header = self.font_small.render(f"{group}/", True, MENU_MUTED)
                self.screen.blit(header, (panel.x + 12, y + 2))
                y += line_height
                continue
            is_current = option == current
            label_image = self.font_mono.render(
                option.label,
                True,
                MENU_ACCENT if is_current else MENU_TEXT,
            )
            size_text = str(option.observation_size) if option.observation_size else "—"
            size_image = self.font_small.render(size_text, True, MENU_MUTED)
            self.screen.blit(label_image, (panel.x + 24, y))
            self.screen.blit(
                size_image,
                (panel.right - 16 - size_image.get_width(), y + 2),
            )
            y += line_height
        if len(lines) > capacity:
            marks = []
            if start > 0:
                marks.append("▲")
            marks.append(f"{current_line + 1}/{len(lines)}")
            if start + capacity < len(lines):
                marks.append("▼")
            status = self.font_small.render(" ".join(marks), True, MENU_MUTED)
            self.screen.blit(
                status,
                (panel.right - 12 - status.get_width(), panel.y + 4),
            )

    def run(self) -> None:
        """Run the dashboard until the window closes."""
        while self.running:
            real_seconds = self.clock.tick(FPS) / 1000.0
            self._handle_events(pygame.event.get())
            now = time.monotonic()
            if now - self._last_reload >= self.reload_seconds:
                self._last_reload = now
                self.reload_policy()
            self.step_frames(real_seconds)
            self.draw_frame()
        self.close()

    def close(self) -> None:
        """Release Pygame resources."""
        pygame.quit()

    @staticmethod
    def _compute_grid(
        panels: int,
        screen_size: tuple[int, int],
        header_height: int = 0,
    ) -> list[pygame.Rect]:
        columns = min(GRID_COLUMNS, panels) if panels > 0 else 1
        rows = (panels + columns - 1) // columns
        area_width = screen_size[0] - METRICS_WIDTH
        cell_width = area_width // columns
        cell_height = (screen_size[1] - header_height) // rows
        rects: list[pygame.Rect] = []
        for index in range(panels):
            row, column = divmod(index, columns)
            rects.append(
                pygame.Rect(
                    column * cell_width,
                    header_height + row * cell_height,
                    cell_width,
                    cell_height,
                )
            )
        return rects

    def draw_frame(self) -> None:
        """Render the header, every panel, and the metrics column."""
        self.screen.fill(theme.BACKGROUND)
        self._draw_header()
        for index, (rect, panel) in enumerate(zip(self._grid, self._panels, strict=True)):
            self._draw_panel(rect, panel, index)
        self._draw_metrics()
        if self._menu_mode is not None:
            self._draw_menu()
        pygame.display.flip()

    def _draw_header(self) -> None:
        bar = pygame.Rect(0, 0, self.screen.get_width(), HEADER_HEIGHT)
        pygame.draw.rect(self.screen, theme.SURFACE, bar)
        pygame.draw.line(
            self.screen,
            theme.BORDER,
            (0, HEADER_HEIGHT - 1),
            (bar.right, HEADER_HEIGHT - 1),
        )
        brand = self.font_sans.render("QB LAB", True, theme.ACCENT)
        self.screen.blit(brand, (theme.SPACE_LG, 13))
        run = self.font_sans.render(self.run_dir.name, True, theme.TEXT)
        self.screen.blit(run, (theme.SPACE_LG + brand.get_width() + 14, 13))
        chip_image = self.font_sans.render(self.mode_label(), True, theme.TEXT)
        chip = pygame.Rect(
            bar.right - chip_image.get_width() - 36,
            10,
            chip_image.get_width() + 20,
            26,
        )
        pygame.draw.rect(self.screen, theme.SURFACE_ALT, chip, border_radius=theme.RADIUS_SM)
        pygame.draw.rect(self.screen, theme.BORDER, chip, 1, border_radius=theme.RADIUS_SM)
        self.screen.blit(chip_image, (chip.x + 10, chip.y + 4))
        status = self.font_mono.render(self.checkpoint_status(), True, theme.TEXT_MUTED)
        self.screen.blit(status, (chip.x - status.get_width() - 16, 14))

    def _draw_panel(
        self,
        rect: pygame.Rect,
        panel: DemoPanel | DriveDemoPanel,
        index: int,
    ) -> None:
        margin = theme.SPACE_SM
        header_height = 30
        pygame.draw.rect(self.screen, theme.SURFACE, rect, border_radius=theme.RADIUS_LG)
        pygame.draw.rect(self.screen, theme.BORDER, rect, 1, border_radius=theme.RADIUS_LG)
        field_rect = pygame.Rect(
            rect.x + margin,
            rect.y + header_height,
            rect.width - 2 * margin,
            rect.height - header_height - margin,
        )
        field_surface = self.screen.subsurface(field_rect)
        view = self._panel_view(field_rect, field_rect.height, index, margin=0)
        state = panel.play.state
        goal_line = 100.0
        if isinstance(panel, (DriveDemoPanel, RoleDriveDemoPanel, CompositionDemoPanel)):
            goal_line = panel.drive.drive_config.goal_line
        draw_field(field_surface, view, goal_line)
        if isinstance(panel, (DriveDemoPanel, RoleDriveDemoPanel, CompositionDemoPanel)):
            drive_state = panel.drive.state
            first_down = min(drive_state.los + drive_state.distance, goal_line)
            draw_scrimmage(field_surface, view, drive_state.los, first_down, goal_line)
        draw_routes(field_surface, panel.receiver_routes, state, view)
        highlight = (
            frozenset({panel.learner_id})
            if isinstance(panel, (RolePlayDemoPanel, RoleDriveDemoPanel))
            else frozenset()
        )
        draw_players(
            field_surface,
            state,
            view,
            frozenset(panel.play.rusher_ids),
            highlight,
        )
        draw_ball(field_surface, state, view)
        if isinstance(panel, CompositionDemoPanel):
            label = panel.last_outcome or panel.down_label
            percentage = 100.0 * panel.touchdowns / panel.drives if panel.drives else 0.0
            stat = f"TD {percentage:0.0f}%"
        elif isinstance(panel, (DriveDemoPanel, RoleDriveDemoPanel)):
            label = panel.last_outcome or panel.down_label
            stat = f"TD {panel.touchdown_pct:0.0f}%"
        elif isinstance(panel, RolePlayDemoPanel):
            label = panel.last_outcome or "in_progress"
            stat = f"{panel.catch_pct:0.0f}%"
        else:
            label = panel.last_outcome or "in_progress"
            stat = f"{panel.completion_pct:0.0f}%"
        label_image = self.font_sans.render(label.replace("_", " "), True, theme.TEXT)
        stat_image = self.font_mono.render(stat, True, theme.ACCENT)
        self.screen.blit(label_image, (rect.x + margin, rect.y + 7))
        self.screen.blit(
            stat_image,
            (rect.right - margin - stat_image.get_width(), rect.y + 7),
        )

    def _metric_rows(self) -> list[tuple[str, str]]:
        if self.env_kind in ("composition", "drive"):
            drives = sum(panel.drives for panel in self._panels)
            touchdowns = sum(panel.touchdowns for panel in self._panels)
            points = sum(panel.points for panel in self._panels)
            first_downs = sum(panel.first_downs for panel in self._panels)
            return [
                ("drives", str(drives)),
                (
                    "TD",
                    f"{touchdowns} "
                    f"({100.0 * touchdowns / drives if drives else 0.0:0.0f}%)",
                ),
                ("puntos/drive", f"{points / drives if drives else 0.0:0.3f}"),
                (
                    "1er downs/drive",
                    f"{first_downs / drives if drives else 0.0:0.2f}",
                ),
            ]
        if self.env_kind == "role_play":
            plays = sum(panel.plays for panel in self._panels)
            catches = sum(panel.catches for panel in self._panels)
            yards = sum(panel.yards for panel in self._panels)
            return [
                ("jugadas", str(plays)),
                (
                    "atrapadas",
                    f"{catches} ({100.0 * catches / plays if plays else 0.0:0.0f}%)",
                ),
                ("yardas/atrapada", f"{yards / catches if catches else 0.0:0.2f}"),
            ]
        if self.env_kind == "role_drive":
            drives = sum(panel.drives for panel in self._panels)
            allowed = sum(panel.touchdowns_allowed for panel in self._panels)
            sacks = sum(panel.sacks for panel in self._panels)
            return [
                ("drives", str(drives)),
                (
                    "TD permitidos",
                    f"{allowed} ({100.0 * allowed / drives if drives else 0.0:0.0f}%)",
                ),
                ("sacks", str(sacks)),
            ]
        finished = sum(panel.finished for panel in self._panels)
        completed = sum(panel.completed for panel in self._panels)
        completion = 100.0 * completed / finished if finished else 0.0
        return [
            ("jugadas", str(finished)),
            ("completados", f"{completion:0.1f}%"),
        ]

    def _controls_hint(self) -> str:
        if self.env_kind == "composition":
            return "M params · N modelos · P pausa · 1/2/3 · R · Esc"
        return "Izq/Der checkpoint · L vivo · P pausa · R · Esc"

    def _draw_metrics(self) -> None:
        rect = pygame.Rect(
            self.screen.get_width() - METRICS_WIDTH,
            HEADER_HEIGHT,
            METRICS_WIDTH,
            self.screen.get_height() - HEADER_HEIGHT,
        )
        pygame.draw.rect(self.screen, theme.BACKGROUND, rect)
        pygame.draw.line(
            self.screen,
            theme.BORDER,
            (rect.x, rect.y),
            (rect.x, rect.bottom),
        )
        progress = read_progress(self.run_dir)
        timesteps = progress[-1][0] if progress else 0
        reward = progress[-1][1] if progress else 0.0
        x = rect.x + theme.SPACE_LG
        width = rect.width - 2 * theme.SPACE_LG
        y = rect.y + theme.SPACE_LG
        y = self._draw_section(
            x,
            y,
            width,
            "RUN",
            (
                ("run", self.run_dir.name),
                ("modo", self.mode_label()),
                ("timesteps", f"{timesteps:,}"),
                ("reward", f"{reward:0.3f}"),
            ),
        )
        y = self._draw_section(x, y, width, "RENDIMIENTO", self._metric_rows())
        chart_height = max(120, rect.bottom - y - 64)
        self._draw_curve(rect, progress, top=y, height=chart_height)
        hint = self.font_small.render(self._controls_hint(), True, theme.TEXT_MUTED)
        self.screen.blit(hint, (x, rect.bottom - 26))

    def _draw_section(
        self,
        x: int,
        y: int,
        width: int,
        title: str,
        rows: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    ) -> int:
        card = pygame.Rect(x, y, width, 32 + len(rows) * 24 + 10)
        pygame.draw.rect(self.screen, theme.SURFACE, card, border_radius=theme.RADIUS_MD)
        pygame.draw.rect(self.screen, theme.BORDER, card, 1, border_radius=theme.RADIUS_MD)
        title_image = self.font_small.render(title, True, theme.TEXT_MUTED)
        self.screen.blit(title_image, (card.x + 12, card.y + 8))
        row_y = card.y + 28
        for label, value in rows:
            label_image = self.font_small.render(label, True, theme.TEXT_MUTED)
            value_image = self.font_mono.render(value, True, theme.TEXT)
            self.screen.blit(label_image, (card.x + 12, row_y))
            self.screen.blit(
                value_image,
                (card.right - 12 - value_image.get_width(), row_y),
            )
            row_y += 24
        return card.bottom + theme.SPACE_MD

    def _draw_curve(
        self,
        rect: pygame.Rect,
        progress: list[tuple[int, float]],
        top: int = 182,
        height: int = 220,
    ) -> None:
        chart = pygame.Rect(
            rect.x + theme.SPACE_LG,
            top,
            rect.width - 2 * theme.SPACE_LG,
            height,
        )
        pygame.draw.rect(self.screen, theme.SURFACE, chart, border_radius=theme.RADIUS_MD)
        pygame.draw.rect(self.screen, theme.BORDER, chart, 1, border_radius=theme.RADIUS_MD)
        title_image = self.font_small.render("RECOMPENSA", True, theme.TEXT_MUTED)
        self.screen.blit(title_image, (chart.x + 12, chart.y + 8))
        if len(progress) < 2:
            empty = self.font_small.render("sin datos", True, theme.TEXT_MUTED)
            self.screen.blit(
                empty,
                (chart.centerx - empty.get_width() // 2, chart.centery - 8),
            )
            return
        plot = pygame.Rect(
            chart.x + 12,
            chart.y + 32,
            chart.width - 24,
            chart.height - 44,
        )
        rewards = [point[1] for point in progress]
        low = min(rewards)
        high = max(rewards)
        span = (high - low) if high > low else 1.0
        for fraction in (0.0, 0.5, 1.0):
            grid_y = plot.bottom - int(plot.height * fraction)
            pygame.draw.line(
                self.screen,
                theme.SURFACE_ALT,
                (plot.x, grid_y),
                (plot.right, grid_y),
                1,
            )
        points = [
            (
                plot.x + int(plot.width * index / (len(rewards) - 1)),
                plot.bottom - int(plot.height * (value - low) / span),
            )
            for index, value in enumerate(rewards)
        ]
        area = pygame.Surface((plot.width, plot.height), pygame.SRCALPHA)
        local = [(px - plot.x, py - plot.y) for px, py in points]
        pygame.draw.polygon(
            area,
            (theme.PRIMARY[0], theme.PRIMARY[1], theme.PRIMARY[2], 46),
            local + [(plot.width, plot.height), (0, plot.height)],
        )
        self.screen.blit(area, plot.topleft)
        pygame.draw.lines(self.screen, theme.PRIMARY_SOFT, False, points, 2)
        pygame.draw.circle(self.screen, theme.ACCENT, points[-1], 4)
        high_image = self.font_small.render(f"{high:0.2f}", True, theme.TEXT_MUTED)
        low_image = self.font_small.render(f"{low:0.2f}", True, theme.TEXT_MUTED)
        self.screen.blit(high_image, (plot.right - high_image.get_width(), plot.y + 2))
        self.screen.blit(low_image, (plot.right - low_image.get_width(), plot.bottom - 18))
