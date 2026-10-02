# M2 — Primer prototipo jugable por un humano: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. TDD obligatorio. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hacer jugable la jugada de pase con 1 QB humano, 2 receptores con rutas y 2 defensivos con cobertura, en Pygame opcional, sin duplicar lógica y manteniendo el modo headless.

**Architecture:** `engine`/`rules` se generalizan a un roster de jugadores con identidad. `ui/controls.py`, `ui/view.py` y `ui/session.py` son puros y testeables sin pantalla; `ui/pygame_app.py` es solo el bucle y el dibujo que consume `PlaySession`. El renderizador lee `PlayState` y nunca muta el motor.

**Tech Stack:** Python 3.11+, NumPy, pytest, Ruff, `pygame-ce` (extra opcional `viz`; `pygame` no tiene wheel para 3.14).

**Spec:** [`AGENTS.md`](../../../AGENTS.md) §5, §7 (M2) y la aprobación del alcance 2 receptores + 2 defensivos.

## Global Constraints

- Ofensiva: 1 QB humano + 2 receptores con rutas predefinidas; defensiva: 2 defensivos con cobertura individual.
- Sin running backs, linieros, formaciones complejas ni reglas de partido.
- La UI nunca contiene lógica de juego: traduce entradas a `Move`/`Throw` y dibuja `PlayState`.
- El proyecto debe instalarse, probarse y ejecutarse headless sin `pygame-ce`.
- Misma secuencia de acciones ⇒ mismos resultados en headless y en la sesión que usa la UI.
- Sin recompensas, Gymnasium ni entrenamiento en M2.

---

### Task 1: Roster 2v2 en engine/rules (headless)

**Files:**
- Modify: `src/qb_sim/engine/entities.py`
- Modify: `src/qb_sim/config.py`
- Modify: `src/qb_sim/rules/play.py`
- Modify: `src/qb_sim/__init__.py`
- Modify: `tests/test_play.py`
- Modify: `README.md`

**Interfaces:**
- `PlayerState` añade `player_id: str = ""`.
- `PlayState` cambia a `players: list[PlayerState]`, `carrier_id: str | None = None` y método `player(player_id) -> PlayerState`.
- `SimulationConfig`: `receiver_starts: tuple[tuple[float, float], ...] = ((20.0, 43.0), (20.0, 10.0))` y `defender_starts: tuple[...] = ((18.0, 43.0), (18.0, 10.0))`; desaparecen `receiver_start` y `defender_start`.
- `Play(config=None, routes=(RouteName.POST, RouteName.SLANT))`; una ruta por receptor.
- IDs: `qb`, `wr1`, `wr2`, `cb1`, `cb2`. `cb1` cubre a `wr1` y `cb2` a `wr2`.
- Contactos de pase: para cada receptor y defensivo se calcula su fracción de tick; gana la menor con prioridad defensa (intercepción) > recepción > fuera de límites > incompleto; los empates se rompen por índice de jugador.
- Tras la recepción, `carrier_id` identifica al portador, corre hacia el fondo rival y cualquier defensivo que lo toque lo tacklea.

- [x] **Step 1: Escribir pruebas del roster (deben fallar)**

```python
def test_reset_builds_two_receivers_and_two_defenders_with_ids() -> None:
    state = Play().reset()

    assert [player.player_id for player in state.players] == [
        "qb",
        "wr1",
        "wr2",
        "cb1",
        "cb2",
    ]
    assert state.player("wr1").role is Role.RECEIVER
    assert state.player("cb2").role is Role.DEFENDER


def test_each_receiver_runs_its_own_route() -> None:
    play = Play()
    state = play.step()

    assert state.player("wr1").position[1] > 43.0  # post: arranca recto
    assert state.player("wr2").position[1] > 10.0  # slant: rompe hacia el centro
```

Run: `python -m pytest tests/test_play.py -q`
Expected: FAIL porque `PlayState` no tiene `players` y `Play` no acepta `routes`.

- [x] **Step 2: Generalizar entidades y configuración**

- `PlayerState.player_id` y `PlayState.player(player_id)`.
- `PlayState.carrier_id` en lugar de `carrier`.
- Validación de `receiver_starts`/`defender_starts`: mismas longitudes ≥ 1 y todos los puntos dentro del campo.
- `Play.reset()` construye `players` en orden determinista y una `Route` por receptor.

- [x] **Step 3: Adaptar reglas y migrar las pruebas existentes**

- `_advance_receivers` avanza cada ruta; `_advance_defenders` persigue al receptor asignado o al portador.
- `_advance_pass` evalúa contactos de todos los receptores/defensivos y resuelve por `(tick_fraction, prioridad, índice)`.
- Migrar pruebas: `receiver_start=(x, y)` → `receiver_starts=((x, y), (119.0, 0.0))`; `defender_start=(x, y)` → `defender_starts=((x, y), (0.0, 0.0))`; `state.receiver` → `state.player("wr1")`; `state.carrier` → `state.carrier_id == "wr1"`.

- [x] **Step 4: Añadir pruebas 2v2 y verificar**

```python
def test_pass_to_second_receiver_uses_its_id_as_carrier() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((119.0, 0.0), (30.0, 10.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    play.step()
    play.throw_to(vec2(30.0, 10.0))

    state = _run_until(play, lambda current: current.ball.status is BallStatus.CARRIED)

    assert state.carrier_id == "wr2"


def test_second_defender_can_tackle_the_carrier() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((119.0, 0.0), (21.5, 26.65)),
        defender_starts=((0.0, 0.0), (23.5, 26.65)),
        defender_speed=8.0,
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.0)
    carried = _run_until(play, lambda current: current.ball.status is BallStatus.CARRIED)
    assert carried.carrier_id == "wr2"

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.COMPLETED
```

Run: `python -m pytest -q && ruff check .`
Expected: todas las pruebas pasan.

---

### Task 2: Sesión y controles headless (sin pygame)

**Files:**
- Create: `src/qb_sim/ui/__init__.py`
- Create: `src/qb_sim/ui/view.py`
- Create: `src/qb_sim/ui/controls.py`
- Create: `src/qb_sim/ui/session.py`
- Create: `tests/test_controls.py`
- Create: `tests/test_session.py`

**Interfaces:**
- `ViewTransform(field, screen_width, screen_height, margin)` con `to_screen(point)` y `to_field(point)`; y si `screen` es la ventana, `to_field` invierte `to_screen`.
- `direction_from_keys(held: frozenset[str]) -> Vector2`: WASD/flechas, diagonales normalizadas, teclas opuestas se cancelan.
- `FrameInput(held, pressed, mouse_field, clicked)`.
- `PlaySession(play=None, routes=...)`: `step_frame(frame) -> PlayState`, `toggle_pause()`, `restart()`, `paused`, `selected_index`.
- Controles: `1`/`2` seleccionan receptor; `Space` lanza al receptor seleccionado con lead calculado; clic izquierdo lanza al punto del ratón; `P` pausa, `R` reinicia.
- La sesión nunca importa pygame.

- [x] **Step 1: Escribir pruebas de vista y teclado (deben fallar)**

```python
def test_direction_from_keys_normalizes_diagonals() -> None:
    direction = direction_from_keys(frozenset({"w", "d"}))
    assert direction.tolist() == pytest.approx([2.0**-0.5, 2.0**-0.5])


def test_direction_from_keys_cancels_opposites() -> None:
    assert direction_from_keys(frozenset({"w", "s"})).tolist() == [0.0, 0.0]


def test_view_transform_round_trips_field_coordinates() -> None:
    view = ViewTransform(FieldBounds(), screen_width=900, screen_height=500, margin=20)
    assert view.to_field(view.to_screen(vec2(60.0, 26.65))).tolist() == pytest.approx([60.0, 26.65])
```

- [x] **Step 2: Implementar `view.py` y `controls.py`**

- `ViewTransform` con escala uniforme, márgenes y eje `y` invertido (pantalla hacia abajo).
- `direction_from_keys` sobre `w/a/s/d` y flechas.

- [x] **Step 3: Escribir pruebas de sesión (deben fallar)**

```python
def test_space_throws_to_selected_receiver_lead() -> None:
    play = Play()
    session = PlaySession(play)
    session.step_frame(
        FrameInput(held=frozenset(), pressed=frozenset({"1"}), mouse_field=None, clicked=False)
    )
    session.step_frame(
        FrameInput(held=frozenset(), pressed=frozenset({"space"}), mouse_field=None, clicked=False)
    )
    receiver = play.state.player("wr1")
    assert play.state.ball.target.tolist() == pytest.approx(receiver.position.tolist())


def test_click_throws_to_mouse_field_point() -> None:
    play = Play()
    session = PlaySession(play)
    session.step_frame(
        FrameInput(
            held=frozenset(), pressed=frozenset(), mouse_field=vec2(40.0, 20.0), clicked=True
        )
    )
    assert play.state.ball.target.tolist() == pytest.approx([40.0, 20.0])


def test_pause_stops_the_clock_and_restart_resets() -> None:
    session = PlaySession(Play())
    session.step_frame(FrameInput(frozenset(), frozenset({"p"}), None, False))
    session.step_frame(FrameInput(frozenset(), frozenset(), None, False))
    assert session.play.state.elapsed_seconds == 0.0
    session.step_frame(FrameInput(frozenset(), frozenset({"r"}), None, False))
    assert session.play.state.outcome is PlayOutcome.IN_PROGRESS
```

- [x] **Step 4: Implementar `PlaySession`**

- `step_frame`: reinicio (`r`), pausa (`p`), selección (`1`/`2`), movimiento con teclas mantenidas, `Space` con lead del receptor seleccionado, clic al punto del ratón; después `play.step(qb_action)`.
- Lead del receptor seleccionado: `distance(qb, receiver) / ball_speed` como estimación de vuelo.

- [x] **Step 5: Verificar y documentar la equivalencia headless**

Añadir a `tests/test_session.py` una prueba que ejecute dos sesiones con la misma lista de `FrameInput` y compare trazas; es la base de la equivalencia con la UI.

Run: `python -m pytest -q && ruff check .`

---

### Task 3: Aplicación Pygame (extra opcional `viz`)

**Files:**
- Modify: `pyproject.toml` (`viz = ["pygame-ce>=2.5"]`)
- Create: `src/qb_sim/ui/rendering.py`
- Create: `src/qb_sim/ui/pygame_app.py`
- Create: `tests/test_pygame_app.py`

**Interfaces:**
- `PygameApp(session: PlaySession, screen_size=(900, 500))` con `run()` y `draw_frame(surface)`.
- `rendering.draw_field(surface, view)`, `draw_players(surface, state, view, selected_index)`, `draw_routes(surface, play, view)`, `draw_ball(surface, state, view)`, `draw_hud(...)`.
- El bucle convierte eventos a `FrameInput`, aplica `step_frame` con acumulador de paso fijo y dibuja; `P` pausa, `R` reinicia.
- `pygame_app` importa `pygame` dentro del módulo; `qb_sim` y `qb_sim.ui.session` no lo importan.

- [x] **Step 1: Declarar el extra e instalar `pygame-ce`**

```bash
python -m pip install -e ".[dev,viz]"
```

- [x] **Step 2: Escribir la prueba de equivalencia con video dummy (debe fallar)**

```python
pygame = pytest.importorskip("pygame")


def test_graphical_loop_matches_headless_session(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    frames = [FrameInput(frozenset({"w"}), frozenset(), None, False) for _ in range(5)]
    headless = PlaySession(Play())
    for frame in frames:
        headless.step_frame(frame)

    app = PygameApp(PlaySession(Play()))
    for frame in frames:
        app.step(frame)

    assert app.session.play.state.elapsed_seconds == headless.play.state.elapsed_seconds
    np.testing.assert_allclose(
        app.session.play.state.player("wr1").position,
        headless.play.state.player("wr1").position,
    )
```

- [x] **Step 3: Implementar `rendering.py` y `pygame_app.py`**

- Campo con líneas de yardas, end zones, jugadores como círculos con color por rol, receptor seleccionado destacado, rutas como polilíneas, balón como punto, HUD con tiempo, resultado, receptor seleccionado y controles.
- `PygameApp.step(frame)` ejecuta un tick de la sesión y `draw_frame` dibuja; `run()` gestiona eventos reales, reloj y acumulador de paso fijo.

- [x] **Step 4: Verificar la app y el modo headless sin pygame**

```bash
python -m pytest
python -c 'import sys, qb_sim; import qb_sim.ui.session; print("pygame" in sys.modules)'
```

Expected: `False` (la sesión headless no importa pygame) y todas las pruebas pasan.

---

### Task 4: Documentación y verificación final

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md` (§11 y decisión abierta nº 2)

- [x] **Step 1: Documentar controles y ejecución**

- `README.md`: controles (WASD/flechas, `1`/`2`, `Space`, clic, `P`, `R`), cómo lanzar el visualizador (`python -m qb_sim.ui.pygame_app`) y cómo jugar headless con `PlaySession`.
- `AGENTS.md` §11: M2 implementado; decisión abierta nº 2 resuelta con 2 receptores + 2 defensivos.

- [x] **Step 2: Verificación completa**

```bash
python -m pytest
ruff check .
ruff format --check .
python -c 'import sys, qb_sim; print([m for m in ("pygame", "gymnasium", "torch") if m in sys.modules])'
```

Expected: todas las pruebas pasan; lint y formato limpios; lista de importaciones prohibidas vacía.

## Self-Review

- **Cobertura:** roster 2v2 (Task 1), controles humanos traducidos a `Move`/`Throw` (Task 2), visualizador opcional sin lógica duplicada (Task 3), equivalencia headless/gráfico (Tasks 2 y 3), documentación y criterios (Task 4).
- **Sin placeholders:** cada tarea incluye pruebas y comandos concretos.
- **Tipos consistentes:** `players`, `carrier_id`, `routes`, `FrameInput`, `ViewTransform`, `PlaySession`, `PygameApp` se definen antes de usarse.
- **Fuera de alcance:** más posiciones, Gymnasium, recompensas, MARL y registro JSONL (M3–M4).
