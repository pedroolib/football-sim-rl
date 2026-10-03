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
from qb_sim.ui.menu import ModelMenu, ParameterMenu
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
        self.font = pygame.font.Font(None, 22)
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
        self._grid = self._compute_grid(panels, screen_size)

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
            for label in ("ppo_drive_qb_v12", "ppo_drive_qb_v11"):
                if self._model_menu.choose("qb", label):
                    break
        defaults = {
            "wr1": ("ppo_wr_v1",),
            "wr2": ("ppo_wr_v1",),
            "cb1": ("ppo_cb_v1",),
            "cb2": ("ppo_cb_v1",),
            "ru1": ("ppo_ru_v1_bc", "ppo_ru_v1"),
        }
        for role, labels in defaults.items():
            for label in labels:
                if self._model_menu.choose(role, label):
                    break

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
        if self.paused:
            return
        self._accumulator += real_seconds * self.speed
        dt = self.scenario.config.dt
        while self._accumulator >= dt:
            for panel in self._panels:
                panel.tick()
            self._accumulator -= dt

    def _handle_events(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self._menu_mode is not None:
                    if event.key == pygame.K_ESCAPE:
                        self._menu_mode = None
                    elif event.key == pygame.K_m and self.env_kind == "composition":
                        self._menu_mode = "params"
                    elif event.key == pygame.K_n and self.env_kind == "composition":
                        self._menu_mode = "models"
                    elif event.key == pygame.K_UP:
                        self._move_menu_selection(-1)
                    elif event.key == pygame.K_DOWN:
                        self._move_menu_selection(1)
                    elif event.key == pygame.K_LEFT:
                        self._adjust_menu(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._adjust_menu(1)
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
        row_height = 24
        visible_rows = 14
        if self._menu_mode == "params" and self._parameter_menu is not None:
            title = "PARÁMETROS  ↑/↓ elegir · ←/→ ajustar · M/Esc cerrar"
            lines = [
                self._parameter_menu.label(index)
                for index in range(len(self._parameter_menu.entries))
            ]
            selected = self._parameter_menu.index
        elif self._model_menu is not None:
            title = "MODELOS  ↑/↓ rol · ←/→ modelo · N/Esc cerrar"
            roles = list(self._model_menu.options)
            lines = [
                f"{role}: {self._model_menu.current(role).label}" for role in roles
            ]
            selected = self._model_menu.index
        else:
            return
        start = max(0, min(selected - visible_rows // 2, max(0, len(lines) - visible_rows)))
        window = lines[start : start + visible_rows]
        height = (len(window) + 2) * row_height + 16
        rect = pygame.Rect(40, 40, 660, height)
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill((10, 14, 18, 235))
        self.screen.blit(overlay, rect.topleft)
        pygame.draw.rect(self.screen, (90, 150, 220), rect, 2)
        title_image = self.font.render(title, True, (255, 235, 150))
        self.screen.blit(title_image, (rect.x + 12, rect.y + 8))
        for row, line in enumerate(window):
            index = start + row
            color = (255, 255, 255) if index == selected else (200, 205, 210)
            image = self.font.render(line, True, color)
            self.screen.blit(image, (rect.x + 24, rect.y + 34 + row * row_height))

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
    def _compute_grid(panels: int, screen_size: tuple[int, int]) -> list[pygame.Rect]:
        columns = min(GRID_COLUMNS, panels) if panels > 0 else 1
        rows = (panels + columns - 1) // columns
        area_width = screen_size[0] - METRICS_WIDTH
        cell_width = area_width // columns
        cell_height = screen_size[1] // rows
        rects: list[pygame.Rect] = []
        for index in range(panels):
            row, column = divmod(index, columns)
            rects.append(
                pygame.Rect(column * cell_width, row * cell_height, cell_width, cell_height)
            )
        return rects

    def draw_frame(self) -> None:
        """Render every panel and the metrics column."""
        self.screen.fill((24, 28, 32))
        for rect, panel in zip(self._grid, self._panels, strict=True):
            self._draw_panel(rect, panel)
            pygame.draw.rect(self.screen, (60, 66, 72), rect, 1)
        self._draw_metrics()
        if self._menu_mode is not None:
            self._draw_menu()
        pygame.display.flip()

    def _draw_panel(self, rect: pygame.Rect, panel: DemoPanel | DriveDemoPanel) -> None:
        margin = 8
        title_height = 26
        field = self.scenario.config.field
        scale = (rect.width - 2 * margin) / field.length_yards
        field_height = int(field.width_yards * scale)
        content_height = field_height + 2 * margin + title_height
        surface = pygame.Surface((rect.width, content_height))
        field_surface = surface.subsurface(
            pygame.Rect(0, title_height, rect.width, content_height - title_height)
        )
        view = ViewTransform(field, rect.width, content_height - title_height, margin=margin)
        state = panel.play.state
        draw_field(field_surface, view)
        if isinstance(panel, (DriveDemoPanel, RoleDriveDemoPanel, CompositionDemoPanel)):
            drive_state = panel.drive.state
            goal_line = panel.drive.drive_config.goal_line
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
            title = self.font.render(
                f"{label}  TD {percentage:0.0f}%",
                True,
                (240, 240, 240),
            )
        elif isinstance(panel, (DriveDemoPanel, RoleDriveDemoPanel)):
            label = panel.last_outcome or panel.down_label
            title = self.font.render(
                f"{label}  TD {panel.touchdown_pct:0.0f}%",
                True,
                (240, 240, 240),
            )
        elif isinstance(panel, RolePlayDemoPanel):
            label = panel.last_outcome or "in_progress"
            title = self.font.render(
                f"{label}  {panel.catch_pct:0.0f}%",
                True,
                (240, 240, 240),
            )
        else:
            label = panel.last_outcome or "in_progress"
            title = self.font.render(
                f"{label}  {panel.completion_pct:0.0f}%",
                True,
                (240, 240, 240),
            )
        surface.blit(title, (10, 4))
        self.screen.blit(surface, (rect.x, rect.y + (rect.height - content_height) // 2))

    def _draw_metrics(self) -> None:
        rect = pygame.Rect(
            self.screen.get_width() - METRICS_WIDTH,
            0,
            METRICS_WIDTH,
            self.screen.get_height(),
        )
        pygame.draw.rect(self.screen, (18, 22, 26), rect)
        progress = read_progress(self.run_dir)
        timesteps = progress[-1][0] if progress else 0
        reward = progress[-1][1] if progress else 0.0
        lines = [
            f"run: {self.run_dir.name}",
            f"modo: {self.mode_label()}",
            f"timesteps: {timesteps:,}",
            f"ep_rew_mean: {reward:0.3f}",
            self.checkpoint_status(),
        ]
        if self.env_kind == "composition":
            drives = sum(panel.drives for panel in self._panels)
            touchdowns = sum(panel.touchdowns for panel in self._panels)
            points = sum(panel.points for panel in self._panels)
            first_downs = sum(panel.first_downs for panel in self._panels)
            lines.extend(
                (
                    f"drives: {drives}",
                    f"TD: {touchdowns} "
                    f"({100.0 * touchdowns / drives if drives else 0.0:0.0f}%)",
                    f"puntos/drive: {points / drives if drives else 0.0:0.3f}",
                    f"1er downs/drive: {first_downs / drives if drives else 0.0:0.2f}",
                )
            )
        elif self.env_kind == "drive":
            drives = sum(panel.drives for panel in self._panels)
            touchdowns = sum(panel.touchdowns for panel in self._panels)
            points = sum(panel.points for panel in self._panels)
            first_downs = sum(panel.first_downs for panel in self._panels)
            lines.extend(
                (
                    f"drives: {drives}",
                    f"touchdowns: {touchdowns} "
                    f"({100.0 * touchdowns / drives if drives else 0.0:0.0f}%)",
                    f"puntos/drive: {points / drives if drives else 0.0:0.3f}",
                    f"1er downs/drive: {first_downs / drives if drives else 0.0:0.2f}",
                )
            )
        elif self.env_kind == "role_play":
            plays = sum(panel.plays for panel in self._panels)
            catches = sum(panel.catches for panel in self._panels)
            yards = sum(panel.yards for panel in self._panels)
            lines.extend(
                (
                    f"jugadas: {plays}",
                    f"atrapadas: {catches} "
                    f"({100.0 * catches / plays if plays else 0.0:0.0f}%)",
                    f"yardas/atrapada: {yards / catches if catches else 0.0:0.2f}",
                )
            )
        elif self.env_kind == "role_drive":
            drives = sum(panel.drives for panel in self._panels)
            allowed = sum(panel.touchdowns_allowed for panel in self._panels)
            sacks = sum(panel.sacks for panel in self._panels)
            lines.extend(
                (
                    f"drives: {drives}",
                    f"TD permitidos: {allowed} "
                    f"({100.0 * allowed / drives if drives else 0.0:0.0f}%)",
                    f"sacks: {sacks}",
                )
            )
        else:
            finished = sum(panel.finished for panel in self._panels)
            completed = sum(panel.completed for panel in self._panels)
            completion = 100.0 * completed / finished if finished else 0.0
            lines.append(f"completados: {completion:0.1f}%  ({finished} jugadas)")
        controls = "←/→ checkpoint | L vivo | P pausa | 1/2/3 | R | Esc"
        if self.env_kind == "composition":
            controls = "M parámetros | N modelos | P pausa | 1/2/3 | R | Esc"
        lines.extend(("", controls))
        for index, line in enumerate(lines):
            image = self.font.render(line, True, (228, 232, 236))
            self.screen.blit(image, (rect.x + 14, 14 + index * 24))
        self._draw_curve(rect, progress, top=14 + len(lines) * 24 + 8)

    def _draw_curve(
        self,
        rect: pygame.Rect,
        progress: list[tuple[int, float]],
        top: int = 182,
    ) -> None:
        chart = pygame.Rect(rect.x + 14, top, rect.width - 28, 220)
        pygame.draw.rect(self.screen, (28, 32, 38), chart, border_radius=6)
        if len(progress) < 2:
            return
        rewards = [point[1] for point in progress]
        low = min(rewards)
        high = max(rewards)
        span = (high - low) if high > low else 1.0
        for index in range(1, len(rewards)):
            x0 = chart.x + chart.width * (index - 1) / (len(rewards) - 1)
            x1 = chart.x + chart.width * index / (len(rewards) - 1)
            y0 = chart.bottom - chart.height * (rewards[index - 1] - low) / span
            y1 = chart.bottom - chart.height * (rewards[index] - low) / span
            pygame.draw.line(self.screen, (120, 200, 255), (x0, y0), (x1, y1), 2)
