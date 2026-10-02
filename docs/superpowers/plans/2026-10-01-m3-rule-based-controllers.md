# M3 — Controladores basados en reglas: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. TDD obligatorio. Steps use checkbox (`- [x]`) syntax.

**Goal:** Formalizar la interfaz de agente por rol, mover rutas/cobertura a controladores de reglas, registrar/reproducir jugadas en JSONL y establecer escenarios y métricas de referencia.

**Architecture:** `Play` queda mecánico: `step(actions)` aplica acciones por jugador y resuelve reglas/física. `agents` contiene la interfaz (`observe`/`act`), observaciones locales por rol y controladores de reglas; `agents/loop.py` orquesta jugadas y series. `recording.py` graba y reproduce; `metrics.py` agrega resultados. Decisiones cerradas: replay JSONL, series de jugadas con métricas, observaciones como dataclasses por rol.

**Tech Stack:** Python 3.11+, NumPy, pytest, Ruff. Sin Gymnasium ni Pygame en la ruta headless.

**Spec:** [`AGENTS.md`](../../../AGENTS.md) §4, §6, §7 (M3) y §12.

## Global Constraints

- Los agentes solo ejecutan acciones permitidas para su rol.
- La misma semilla, configuración, estado inicial y secuencia de acciones producen una traza idéntica.
- El registro debe cubrir acciones humanas y de agentes por igual.
- La UI no contiene lógica de juego: usa los mismos agentes y acciones.
- Sin reglas de downs, reloj ni anotaciones (M8).

---

### Task 1: Interfaz de agente, observaciones y acciones por rol

**Files:** `engine/actions.py`, `agents/__init__.py`, `agents/observations.py`, `agents/base.py`, `tests/test_agents.py`.

**Interfaces:**
- `allowed_actions(role) -> tuple[type, ...]` en `engine/actions.py` (QB: `Move`/`Throw`; receptor/defensivo: `Move`).
- Observaciones congeladas por rol: `ReceiverObservation`, `DefenderObservation`, `QuarterbackObservation` con posición/velocidad propias, `time_step`, balón y solo información local.
- `BaseAgent` (ABC): `player_id`, `role`, `reset()`, `observe(state)`, `act(observation)`, `decide(state) = act(observe(state))`.

- [x] Pruebas: `allowed_actions` por rol; observaciones no incluyen información de otros roles; `decide` usa `observe`+`act`.

### Task 2: Controladores de reglas

**Files:** `rules/routes.py` (acceso público al waypoint activo), `agents/rule_controllers.py`, `tests/test_rule_controllers.py`.

**Interfaces:**
- `ReceiverRouteController(player_id, route, speed)`: avanza la ruta con `Route.advance` y devuelve `Move` equivalente.
- `ManCoverageController(player_id, speed, assignment_id)`: persigue al receptor asignado o al portador.
- `HeuristicQuarterback(player_id, config, open_threshold, throw_after_seconds)`: lanza al receptor más abierto cuando hay ventana o al agotarse el tiempo; corre alejándose de la presión; una sola decisión por tick.

- [x] Pruebas: ruta recorrida sin sobrepasar waypoints; cobertura cambia al portador tras la recepción; QB lanza con receptor abierto, no antes si está cubierto y sí al vencer el plazo; presión produce movimiento de escape.

### Task 3: `Play` mecánico con acciones por jugador

**Files:** `rules/play.py`, `tests/test_play.py`, `tests/test_accuracy.py`.

**Interfaces:**
- `Play(config=None, seed=0)`; `step(actions: Mapping[str, Move | Throw] | None)`.
- Validación: id desconocido o acción no permitida ⇒ `ValueError`; jugador sin acción queda quieto; el balón sigue al portador; tackle/fuera de límites se resuelven en reglas.
- `throw_to`/`throw_with_lead` siguen como atajos del QB.

- [x] Pruebas: acción por jugador aplicada con la velocidad de su rol; rechazo por rol e id; migración de la suite con agentes de reglas.

### Task 4: Loop y escenarios reproducibles

**Files:** `agents/loop.py`, `rules/scenarios.py`, `tests/test_loop.py`, `tests/test_scenarios.py`.

**Interfaces:**
- `build_rule_agents(play, routes, config, include_quarterback=True)`.
- `run_play(scenario=None, controllers=None, seed=None, recorder=None) -> PlayState`.
- `run_series(scenario, count, seed_start=0) -> list[PlayState]`.
- `Scenario(name, config, receiver_routes, seed)` con `default`, `pressured`, `deep`; `get_scenario(name)`.

- [x] Pruebas: dos series con la misma semilla dan los mismos resultados; escenarios válidos; los agentes se resetean por jugada.

### Task 5: Registro y reproducción JSONL

**Files:** `recording.py`, `tests/test_recording.py`.

**Interfaces:**
- `PlayRecorder(path | stream)`: cabecera con semilla, config y escenario; una línea por tick con acciones por jugador; línea final con resultado.
- `replay(record) -> PlayState` reconstruye con la misma semilla y acciones y verifica traza idéntica; error explícito si diverge.

- [x] Pruebas: round-trip idéntico; replay detecta divergencia; el archivo es JSONL legible con acciones humanas y de agentes.

### Task 6: Métricas y baseline por reglas

**Files:** `metrics.py`, `scripts/rule_baseline_report.py`, `tests/test_metrics.py`.

**Interfaces:**
- `summarize(results) -> dict` con: % completos, % intercepción, % incompleto, yardas por intento, yardas por completado, tiempo hasta el lanzamiento y duración media.
- Script reproducible sobre los escenarios con semillas fijas.

- [x] Pruebas: métricas calculadas sobre trazas sintéticas conocidas; reporte reproducible.

### Task 7: Integración UI (humano + agentes)

**Files:** `ui/session.py`, `ui/rendering.py`, `ui/pygame_app.py`, `tests/test_session.py`, `tests/test_pygame_app.py`.

**Interfaces:**
- `PlaySession` construye agentes de reglas para receptores/defensivos y deja el QB al humano; `routes` para el render; `restart` reconstruye agentes.
- `draw_routes(surface, routes, state, view)` recibe las rutas de la sesión.

- [x] Pruebas: equivalencia gráfico/headless con agentes; el humano solo controla al QB.

### Task 8: Documentación y verificación

**Files:** `README.md`, `AGENTS.md`.

- [x] README: agentes, escenarios, replay JSONL y métricas; `AGENTS.md` §11/§12 actualizados (replay resuelto).
- [x] `python -m pytest`, `ruff check .`, `ruff format --check .`, importaciones headless limpias.
