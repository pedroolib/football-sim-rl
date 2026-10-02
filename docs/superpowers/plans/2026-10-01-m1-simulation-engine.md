# M1 — Motor de simulación: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. TDD obligatorio: primero la prueba que falla, después la implementación mínima.

**Goal:** Reestructurar el prototipo en capas `engine`/`rules`, añadir el esquema de acciones híbrido y la regla de tackle, manteniendo la simulación determinista y headless.

**Architecture:** `engine` (geometría, entidades, acciones, física) no conoce reglas ni UI. `rules` (rutas y jugada) consume `engine`. La jugada se avanza con `Play.step(qb_action)` y resuelve eventos por fracción de tick. Los receptores y defensivos usan controladores provisionales por guion hasta M3.

**Tech Stack:** Python 3.11+, NumPy, pytest, Ruff. Sin Gymnasium ni Pygame en M1.

**Spec:** [`AGENTS.md`](../../../AGENTS.md), secciones 4, 5, 7 (M1) y 12 (decisiones aprobadas: acciones híbridas, tackle de toque, pase 2D, replay JSONL para M3).

## Global Constraints

- Determinismo con la misma configuración y secuencia de acciones.
- El motor no importa entrenamiento ni UI.
- Un QB, un receptor, un defensivo y un balón; rutas `go`, `slant`, `post`.
- Regla de tackle V0.1: contacto/toque dentro de `tackle_radius`.
- El pase vuela en 2D a velocidad constante hacia un punto objetivo.
- Sin recompensas, observaciones RL, renderizado ni registro en M1 (el registro llega en M3).
- Type hints, pytest y Ruff en todo el código.

---

### Task 1: Reestructurar el paquete en `engine` y `rules`

Refactor mecánico sin cambio de comportamiento: la suite existente (34 pruebas) debe seguir en verde.

**Files:**
- Create: `src/qb_sim/engine/__init__.py`
- Create: `src/qb_sim/rules/__init__.py`
- Move: `src/qb_sim/geometry.py` → `src/qb_sim/engine/geometry.py`
- Move: `src/qb_sim/entities.py` → `src/qb_sim/engine/entities.py`
- Move: `src/qb_sim/physics.py` → `src/qb_sim/engine/physics.py`
- Move: `src/qb_sim/routes.py` → `src/qb_sim/rules/routes.py`
- Move: `src/qb_sim/simulation.py` → `src/qb_sim/rules/play.py`
- Modify: `src/qb_sim/config.py`, `src/qb_sim/__init__.py`
- Modify: `tests/test_geometry.py`, `tests/test_physics.py`, `tests/test_routes.py`
- Move + Modify: `tests/test_simulation.py` → `tests/test_play.py`

**Interfaces:**
- `qb_sim.engine.geometry` sigue exportando `Vector2`, `vec2`, `distance`, `segment_circle_contact_fraction`, `FieldBounds`.
- `qb_sim.engine.entities` sigue exportando `PlayerState`, `BallState`, `BallStatus`, `PlayOutcome`, `PlayState`.
- `qb_sim.engine.physics` sigue exportando `move_toward`, `moving_circle_contact_fraction`, `segment_field_exit_fraction`, `advance_ball`.
- `qb_sim.rules.routes` exporta `RouteName`, `Route`, `build_route`.
- `qb_sim.rules.play.Play` sustituye a `Simulation` (misma API pública en esta tarea).
- `qb_sim.config.SimulationConfig` permanece en la raíz porque configura motor y reglas.

- [x] **Step 1: Mover archivos y crear los paquetes de capa**

```bash
mkdir -p src/qb_sim/engine src/qb_sim/rules
git mv src/qb_sim/geometry.py src/qb_sim/engine/geometry.py
git mv src/qb_sim/entities.py src/qb_sim/engine/entities.py
git mv src/qb_sim/physics.py src/qb_sim/engine/physics.py
git mv src/qb_sim/routes.py src/qb_sim/rules/routes.py
git mv src/qb_sim/simulation.py src/qb_sim/rules/play.py
git mv tests/test_simulation.py tests/test_play.py
printf '"""Motor de simulación: geometría, entidades, física y acciones."""\n' > src/qb_sim/engine/__init__.py
printf '"""Reglas de la jugada y escenarios."""\n' > src/qb_sim/rules/__init__.py
```

- [x] **Step 2: Actualizar imports**

Reglas de importación:
- `engine/entities.py`: `from qb_sim.engine.geometry import Vector2`
- `engine/physics.py`: `from qb_sim.engine.entities import BallState, BallStatus` y `from qb_sim.engine.geometry import ...`
- `rules/routes.py`: `from qb_sim.config import SimulationConfig`, `from qb_sim.engine.geometry import FieldBounds, Vector2, vec2`
- `rules/play.py`: `from qb_sim.engine.*` y `from qb_sim.rules.routes import ...`
- `config.py`: `from qb_sim.engine.geometry import FieldBounds`
- `qb_sim/__init__.py`: reexporta `Play` desde `qb_sim.rules.play` y el resto desde `qb_sim.engine.*` / `qb_sim.rules.routes`.
- Tests: `from qb_sim.engine.geometry import ...`, `from qb_sim.engine.physics import ...`, `from qb_sim.rules.play import Play`, `from qb_sim.rules.routes import ...`. En `test_play.py` reemplazar `Simulation` por `Play`.
- Eliminar alias `simulation.py`; no dejar módulos duplicados.

- [x] **Step 3: Verificar que la suite sigue verde a nivel de comportamiento**

Run: `python -m pytest -q && ruff check . && ruff format --check .`
Expected: 34 passed, sin errores de lint ni formato.

- [ ] **Step 4: Commit del refactor**

```bash
git add -A
git commit -m "refactor: split simulation package into engine and rules layers"
```

---

### Task 2: Esquema de acciones híbrido

**Files:**
- Create: `src/qb_sim/engine/actions.py`
- Create: `tests/test_actions.py`
- Modify: `src/qb_sim/config.py` (añadir `quarterback_speed`)
- Modify: `src/qb_sim/rules/play.py` (paso con acción del QB)
- Modify: `tests/test_play.py`

**Interfaces:**
- `Move(direction: Vector2)`: dirección finita de magnitud ≤ 1 (magnitud 0 = quedarse).
- `Throw(target: Vector2)`: punto objetivo del pase en coordenadas de campo.
- `Play.step(qb_action: Move | Throw | None = None) -> PlayState`: aplica la acción del QB al inicio del tick y avanza el mundo. Las acciones no persisten entre ticks.
- `Play.throw_to(target)` y `Play.throw_with_lead(lead_seconds)` siguen existiendo como atajos que lanzan de inmediato usando el mismo camino interno que `Throw`.
- `Role` en `engine/entities.py` (`QUARTERBACK`, `RECEIVER`, `DEFENDER`) y `PlayerState.role: Role | None = None`.

- [x] **Step 1: Escribir pruebas de validación de acciones (deben fallar)**

```python
import pytest
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.geometry import vec2


def test_move_rejects_direction_above_unit_magnitude() -> None:
    with pytest.raises(ValueError):
        Move(direction=vec2(1.0, 1.0))


def test_move_rejects_non_finite_direction() -> None:
    with pytest.raises(ValueError):
        Move(direction=vec2(float("nan"), 0.0))


def test_move_accepts_zero_and_unit_directions() -> None:
    assert Move(direction=vec2(0.0, 0.0)).direction.tolist() == [0.0, 0.0]
    assert Move(direction=vec2(1.0, 0.0)).direction.tolist() == [1.0, 0.0]


def test_throw_rejects_non_finite_target() -> None:
    with pytest.raises(ValueError):
        Throw(target=vec2(0.0, float("inf")))
```

Run: `python -m pytest tests/test_actions.py -q`
Expected: FAIL por `ModuleNotFoundError: qb_sim.engine.actions`.

- [x] **Step 2: Implementar `engine/actions.py`**

Dos dataclasses congeladas con `__post_init__` que valide finitud, forma `(2,)` y magnitud. `Move` debe copiar el vector para no aliarse con el llamador.

- [x] **Step 3: Escribir pruebas de integración de acciones en `Play` (deben fallar)**

```python
def test_qb_move_action_displaces_him_at_quarterback_speed() -> None:
    play = Play()
    state = play.step(qb_action=Move(direction=vec2(1.0, 0.0)))
    assert state.quarterback.position.tolist() == pytest.approx([20.0 + 6.5 * 0.05, 26.65])


def test_qb_throw_action_launches_ball_toward_target() -> None:
    play = Play()
    state = play.step(qb_action=Throw(target=vec2(40.0, 26.65)))
    assert state.ball.status is BallStatus.IN_FLIGHT
    assert state.ball.target.tolist() == pytest.approx([40.0, 26.65])


def test_actions_do_not_persist_between_ticks() -> None:
    play = Play()
    play.step(qb_action=Move(direction=vec2(1.0, 0.0)))
    first_x = play.state.quarterback.position[0]
    play.step()
    assert play.state.quarterback.position[0] == pytest.approx(first_x)
```

Run: `python -m pytest tests/test_play.py -q`
Expected: FAIL porque `step` no acepta `qb_action` y `Move`/`Throw` no existen.

- [x] **Step 4: Implementar el paso con acción y adaptar los atajos**

En `rules/play.py`:
- `step(qb_action=None)`: si hay `Move`, desplazar al QB `direction * quarterback_speed * dt` antes de avanzar a los demás; si hay `Throw`, lanzar al objetivo.
- Extraer `_launch_throw(target)` para que `throw_to`, `throw_with_lead` y `Throw` compartan validación y lógica.
- Mantener las validaciones de un solo lanzamiento y de fin de jugada.
- Añadir `quarterback_speed: float = 6.5` a `SimulationConfig` con la validación de valores positivos ya existente.

- [x] **Step 5: Verificar** (commit pendiente de tu aprobación)

Run: `python -m pytest -q && ruff check .`
Expected: todas las pruebas pasan.

```bash
git add -A
git commit -m "feat: add hybrid engine action schema and quarterback actions"
```

---

### Task 3: Fase de portador y regla de tackle

**Files:**
- Modify: `src/qb_sim/engine/entities.py` (`BallStatus.CARRIED`, `PlayState.carrier`)
- Modify: `src/qb_sim/config.py` (`tackle_radius`)
- Modify: `src/qb_sim/rules/play.py`
- Modify: `tests/test_play.py`
- Modify: `tests/test_physics.py` si hace falta reutilizar detección de contacto entre jugadores

**Interfaces:**
- Después de una recepción, `ball.status` pasa a `CARRIED`, `state.carrier` es el `Role` del portador, y la jugada continúa.
- El portador avanza `direction = vec2(1.0, 0.0)` a `receiver_speed`; el defensivo persigue al portador.
- Si el defensivo queda a distancia ≤ `tackle_radius` del portador, termina la jugada con `PlayOutcome.COMPLETED` (toque).
- Si el portador sale del campo, termina con `PlayOutcome.COMPLETED`.
- Si se agota el tiempo con una recepción previa, termina con `PlayOutcome.COMPLETED`; sin recepción, `PlayOutcome.TIME_LIMIT`.
- La intercepción sigue terminando la jugada de inmediato.

- [x] **Step 1: Escribir pruebas de la fase de portador (deben fallar)**

```python
def test_catch_switches_ball_to_carried_and_play_continues() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_start=(30.0, 26.65),
        defender_start=(0.0, 0.0),
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.4)
    state = _run_until(lambda: play.state.ball.status is BallStatus.CARRIED)
    assert state.carrier is Role.RECEIVER
    assert state.outcome is PlayOutcome.IN_PROGRESS


def test_tackle_ends_the_play_after_a_completion() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_start=(25.0, 26.65),
        defender_start=(25.0, 28.0),
        receiver_speed=7.0,
        defender_speed=7.5,
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.2)
    state = _run_until_terminal(play)
    assert state.outcome is PlayOutcome.COMPLETED


def test_carrier_out_of_bounds_ends_the_play() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(117.0, 26.65),
        receiver_start=(118.0, 26.65),
        defender_start=(0.0, 0.0),
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.1)
    state = _run_until_terminal(play)
    assert state.outcome is PlayOutcome.COMPLETED
```

Run: `python -m pytest tests/test_play.py -q`
Expected: FAIL porque la recepción todavía termina la jugada y no existe `carrier`.

- [x] **Step 2: Implementar la fase de portador y el tackle**

- Añadir `Role` y `BallStatus.CARRIED`; inicializar roles de los tres jugadores en `reset`.
- Al resolver una recepción dentro de `_advance_pass`, en lugar de terminar: mover el balón al portador, marcar `CARRIED`, fijar `carrier = Role.RECEIVER` y continuar.
- En cada tick con portador: avanzar al portador hacia el fondo rival y al defensivo hacia el portador; comprobar tackle (`distance ≤ tackle_radius`, prioridad sobre fuera de límites), salida del campo y tiempo.
- Añadir `tackle_radius: float = 1.0` a `SimulationConfig`.
- Reutilizar la resolución por fracción de tick cuando el tackle ocurra en el mismo tick que otros eventos; si el contacto empieza dentro del radio, el tackle gana.

- [x] **Step 3: Verificar** (commit pendiente de tu aprobación)

Run: `python -m pytest -q && ruff check .`
Expected: todas las pruebas pasan.

```bash
git add -A
git commit -m "feat: add ball-carrier phase and touch tackle rule"
```

---

### Task 4: Documentación, determinismo y verificación final

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md` (sección 11, estado del repositorio)
- Modify: `tests/test_play.py`

- [x] **Step 1: Probar determinismo con acciones explícitas**

```python
def test_same_config_and_action_sequence_produce_identical_traces() -> None:
    first, second = Play(), Play()
    actions = [Move(direction=vec2(1.0, 0.0)), None, Throw(target=vec2(50.0, 26.65))]
    for action in actions:
        for _ in range(20):
            first.step(qb_action=action)
            second.step(qb_action=action)
    assert first.state.outcome is second.state.outcome
    np.testing.assert_allclose(first.state.receiver.position, second.state.receiver.position)
    np.testing.assert_allclose(first.state.ball.position, second.state.ball.position)
```

Run: `python -m pytest tests/test_play.py -q`
Expected: FAIL si la jugada conserva estado oculto compartido; en caso contrario, pasa y confirma el contrato.

- [x] **Step 2: Actualizar documentación**

- `README.md`: API pública (`Play`, `Move`, `Throw`), regla de tackle, fases de la jugada y ejemplo headless con `step(qb_action=...)`.
- `AGENTS.md` §11: indicar que M1 está implementado en capas `engine`/`rules`, con esquema de acciones híbrido y regla de tackle de toque.

- [x] **Step 3: Verificación completa**

```bash
python -m pytest
ruff check .
ruff format --check .
python -c 'import sys, qb_sim; print([m for m in ("gymnasium", "pygame", "torch") if m in sys.modules])'
```

Expected: todas las pruebas pasan, lint y formato limpios, y la lista de importaciones prohibidas vacía.

- [ ] **Step 4: Commit final del hito**

```bash
git add -A
git commit -m "docs: describe M1 engine/rules API and mark milestone state"
```

## Self-Review (hecha antes de publicar el plan)

- **Cobertura del spec:** campo/entidades/balón (ya existen), movimiento y colisiones (Task 2), paso fijo (existente), pase/recepción/intercepción (existentes, preservadas), tackle y finalización (Task 3), acciones independientes de la UI (Task 2), determinismo (Tasks 1 y 4), capas `engine`/`rules` (Task 1).
- **Sin placeholders:** cada tarea tiene pasos con código o comandos concretos.
- **Tipos consistentes:** `Play`, `Move`, `Throw`, `Role`, `BallStatus.CARRIED`, `PlayState.carrier` y `SimulationConfig.tackle_radius` se definen antes de usarse.
- **Fuera de alcance, por diseño:** interface de agente, recompensas, Gymnasium, Pygame, registro JSONL y `pygame-ce` pertenecen a M2–M4.
