# QB Lab — Instrucciones para Agentes

## 1. Visión del proyecto

Construir un juego 2D de fútbol americano con estilo arcade, inspirado en la experiencia de juego de Retro Bowl, con una diferencia fundamental: **cada jugador del campo terminará controlado por un agente autónomo**, entrenado progresivamente con Reinforcement Learning (RL).

El proyecto tiene dos objetivos complementarios:

1. **Juego jugable por humanos:** desde etapas tempranas debe poder jugarse con controles intuitivos y una experiencia arcade.
2. **Entorno de investigación RL / Multi-Agent RL (MARL):** el mismo motor debe permitir entrenar, evaluar, visualizar y comparar agentes individuales y equipos completos.

La inspiración se limita a las mecánicas generales y al estilo de juego. Código, gráficos, nombres, interfaz y recursos deben ser **originales**; no copiar assets ni contenido propietario de Retro Bowl.

El objetivo final no es un simulador donde solo el quarterback tenga IA, sino un juego donde jugadores ofensivos y defensivos tomen decisiones, se coordinen y aprendan. La implementación debe ser incremental: primero un prototipo sencillo, después equipos completos.

## 2. Filosofía de desarrollo

- Construir incrementalmente; nunca implementar el roadmap completo en un paso.
- Priorizar un sistema pequeño, correcto, testeable y medible sobre uno grande e incompleto.
- Mantener separados el motor de simulación, las reglas del juego, las políticas de los agentes, la interfaz y el entrenamiento.
- El simulador debe funcionar sin interfaz gráfica (*headless*); el entrenamiento nunca depende de la UI.
- El juego humano, los controladores basados en reglas y las políticas entrenadas deben usar el mismo motor y la misma interfaz de acciones.
- Garantizar reproducibilidad: semillas, escenarios configurables y registro de acciones.
- Usar dependencias mínimas; no agregar frameworks complejos antes de que un hito los necesite.
- Usar observaciones locales o parciales cuando corresponda; un agente no debe ver información que un jugador real no tendría.
- Diseñar recompensas que favorezcan el resultado colectivo sin impedir el análisis individual.
- Explicar las decisiones técnicas importantes en lenguaje claro y accesible.
- Comunicarse con el usuario en español; usar inglés para identificadores y terminología técnica estándar.
- No hacer cambios arquitectónicos importantes en silencio. Si un requisito es ambiguo, preguntar antes de implementar.
- No ocultar pruebas fallidas ni debilitar aserciones para hacer pasar una build.

## 3. Tecnología

Base inicial; revisar y justificar cuando un hito lo requiera:

- Python 3.11+ (el entorno actual usa 3.14; verificar compatibilidad de *wheels*).
- NumPy para cálculo numérico.
- Gymnasium para el entorno de un solo agente (M4).
- PyTorch para redes neuronales.
- Stable-Baselines3 para el primer baseline PPO (M5). No asumir que SB3 cubrirá MARL.
- Pygame como dependencia **opcional** de visualización interactiva (M2). El proyecto debe instalarse, probarse y entrenar sin Pygame. Si no hay wheel para la versión de Python, usar `pygame-ce` (compatible a nivel de API).
- PettingZoo u otra interfaz multiagente: **evaluar en M6**; no integrar antes.
- pytest para pruebas y Ruff para formato y lint.

Reglas:

- La simulación no debe importar el framework de entrenamiento ni la UI.
- No introducir Unity, visión por computadora, LLMs externos ni entrenamiento distribuido sin justificación explícita.
- Antes de adoptar una herramienta de MARL, investigar y justificar la decisión con experimentos o referencias.

## 4. Arquitectura por capas

Dependencias permitidas: `engine` no depende de nadie; `rules` depende de `engine`; `agents` dependen de `engine` y `rules`; `ui` y `training` dependen de las capas anteriores, nunca al revés.

### Motor (`engine`)

- Campo y geometría 2D.
- Estado de jugadores y balón; posiciones y velocidades.
- Movimiento, colisiones, tiempo con paso fijo y azar con semilla.
- No conoce reglas de puntuación, interfaz ni frameworks de RL.

### Reglas (`rules`)

- Inicio y finalización de jugada.
- Pase, recepción, intercepción, tackle y salidas del campo.
- Condiciones de anotación, downs y reloj cuando el hito lo requiera.
- Usa el motor; no contiene física duplicada.

### Agentes (`agents`)

- Interfaz común por rol: `observe(state) -> Observation`, `act(observation) -> Action`.
- Los controladores basados en reglas y las políticas entrenadas implementan la misma interfaz y son intercambiables.
- Permite compartir pesos entre jugadores del mismo rol; no entrenar una red independiente por jugador.
- Cada jugador del campo tiene una instancia de agente que decide según su rol y su observación.
- Diseñar desde temprano para pasar de un agente a múltiples agentes sin reescribir el motor.

### Interfaz (`ui`)

- Pygame como capa de visualización interactiva.
- Lee el estado y traduce entradas humanas a los **mismos tipos de acción** que usan los agentes.
- Nunca contiene una implementación independiente de la lógica del juego.
- Debe poder pausar, reiniciar y reproducir jugadas.

### Entrenamiento (`training`)

- Entornos Gymnasium (M4) y multiagente (M6).
- Configuración, checkpoints, evaluación con semillas separadas, métricas y logging.
- No contiene física ni reglas; solo consume el entorno.

### Registro y reproducción

- Registrar semilla, configuración, observaciones, acciones y resultado por jugada.
- Una jugada registrada debe poder reproducirse de forma determinista.

## 5. Jugabilidad humana

El modo humano se desarrolla temprano (M2), no al final como tarea de visualización. La experiencia debe evolucionar hacia un juego arcade.

Controles objetivo, implementados según las mecánicas existentes en cada etapa:

- Movimiento y carrera del jugador controlado.
- Apuntar y lanzar pases.
- Seleccionar jugadas y objetivos.
- Controlar o cambiar al jugador activo cuando las mecánicas lo requieran.
- Pausar, reiniciar y reproducir jugadas.

Reglas:

- No implementar todos los controles en el primer prototipo.
- Las entradas humanas se convierten a acciones válidas del motor.
- A medida que existan agentes para otros roles, el modo humano debe poder usarlos como compañeros y oponentes.

## 6. Agentes por posición

Introducir roles gradualmente. Para cada rol nuevo definir: observación, espacio de acciones, política inicial basada en reglas, recompensas y métricas relevantes, y pruebas de comportamiento.

Objetivos futuros (no requisitos del primer prototipo):

- **Quarterback:** moverse, decidir cuándo lanzar, elegir receptor y objetivo, correr, entregar el balón.
- **Receptores:** ejecutar rutas, ajustar movimiento, buscar espacios, atrapar pases, bloquear.
- **Running backs:** correr con el balón, elegir rutas, esquivar defensivos, bloquear.
- **Defensivos:** cubrir receptores, perseguir al portador, interceptar, presionar al quarterback, tacklear.
- **Línea:** bloquear, proteger espacios y ejecutar acciones de su posición.

No implementar todas las posiciones a la vez.

## 7. Roadmap

Cada hito debe ser verificable y dejar el sistema funcionando. No avanzar sin cumplir los criterios de aceptación del hito actual.

### M0 — Inspección y diseño (en curso)

- Inspeccionar el repositorio y el código existente.
- Identificar qué partes del plan original siguen siendo útiles.
- Definir la arquitectura que soporte juego humano y entrenamiento multiagente.
- Definir las mecánicas mínimas del primer prototipo.
- No implementar código todavía.

### M1 — Motor de simulación

- Campo 2D y modelo básico de jugadores y balón.
- Movimiento, colisiones y avance del tiempo con paso fijo.
- Reglas fundamentales de una jugada: inicio, pase, recepción, intercepción, tackle y finalización.
- Estados y acciones independientes de la interfaz.
- Pruebas de determinismo y de reglas básicas.

### M2 — Primer prototipo jugable por un humano

- Visualizador interactivo con gráficos simples.
- Control del quarterback con teclado y mouse según las mecánicas existentes.
- Receptores y defensivos con comportamientos básicos basados en reglas.
- Jugar, pausar, reiniciar y observar jugadas.
- Validar que el juego se sienta razonable antes de entrenar políticas.
- Mantener el modo headless funcional.

### M3 — Controladores basados en reglas

- Formalizar políticas simples para los roles existentes, con la interfaz común de agentes.
- Permitir jugadas y partidos de prueba con equipos controlados por reglas.
- Registrar acciones, observaciones y resultados.
- Establecer escenarios reproducibles y métricas de referencia.

### M4 — Entorno de entrenamiento de un solo agente

- Entorno compatible con Gymnasium para entrenar inicialmente al quarterback.
- Los demás jugadores usan políticas basadas en reglas.
- Definir observaciones, acciones, recompensas y condiciones de terminación.
- Entrenar y evaluar sin gráficos.
- Elegir una primera tarea pequeña y medible, sin intentar resolver todo el fútbol americano.

### M5 — Entrenamiento y evaluación del quarterback

- Implementar PPO u otro algoritmo apropiado para el primer agente.
- Comparar la política entrenada con los controladores por reglas y con el desempeño humano.
- Mostrar visualmente las jugadas del agente.
- Medir con métricas objetivas y escenarios repetibles.
- Evitar recompensas que produzcan comportamientos sin sentido dentro del juego.

### M6 — Entorno multiagente

- Interfaz multiagente para que varios jugadores decidan durante una misma jugada.
- Evaluar PettingZoo u otra interfaz adecuada.
- Observaciones y acciones específicas por rol.
- Políticas compartidas entre jugadores del mismo rol cuando convenga.
- Empezar con escenarios pequeños (1 contra 1 y grupos reducidos) antes de escalar.
- Mantener compatibilidad con controladores por reglas y modo humano.

### M7 — Aprendizaje y coordinación multiagente

- Investigar e implementar un enfoque adecuado de MARL, como MAPPO, si la arquitectura lo justifica.
- Explorar recompensas compartidas, recompensas por rol y entrenamiento centralizado con ejecución descentralizada.
- Entrenar progresivamente receptores, corredores y defensivos.
- Evaluar coordinación, cobertura, rutas, toma de decisiones y resultados colectivos.
- Incorporar self-play cuando existan políticas suficientemente estables.
- Comparar con las políticas basadas en reglas y versiones anteriores de los agentes.

### M8 — Juego completo inspirado en Retro Bowl

- Expandir el motor a partidos más completos, con equipos y posiciones adicionales.
- Integrar reglas arcade consistentes: downs, yardas por avanzar, anotaciones, posesiones, reloj y selección de jugadas, según el alcance que se defina.
- Permitir partidos entre equipos de agentes, equipos de reglas y modo humano.
- Lograr que todos los jugadores activos sean controlados por agentes en el modo autónomo.
- Interfaz más pulida, estadísticas y herramientas de reproducción.
- Temporadas, plantillas y gestión de equipos son una extensión separada, no un requisito para validar el aprendizaje multiagente.

## 8. Pruebas y criterios de aceptación

Cada hito debe incluir criterios verificables y pruebas automatizadas cuando corresponda. Como mínimo:

- El motor es determinista con la misma semilla y la misma secuencia de acciones.
- La interfaz gráfica y el modo headless producen los mismos resultados para las mismas acciones.
- Los controles humanos se traducen a acciones válidas del motor.
- Los agentes solo ejecutan acciones disponibles para su rol.
- El entorno maneja correctamente terminaciones, truncamientos y recompensas.
- Es posible registrar y reproducir una jugada.
- Las métricas permiten comparar políticas bajo escenarios equivalentes.
- El proyecto no se considera terminado si solo el quarterback tiene una política entrenada y el resto del equipo sigue controlado por reglas.

## 9. Git y gestión de experimentos

- Usar Git con commits enfocados.
- Mantener configuraciones y resultados reproducibles.
- No commitear: entornos virtuales, datasets temporales grandes, checkpoints innecesarios ni cachés generadas.
- Mantener un `.gitignore` y documentar cómo reproducir los experimentos importantes.
- Usar nombres de experimento claros y conservar la configuración asociada a cada checkpoint.

## 10. Comunicación y flujo de trabajo

Al inicio de cada sesión de desarrollo:

1. Leer este archivo.
2. Inspeccionar el repositorio y el código relevante.
3. Identificar el hito actual y el estado del proyecto.
4. Resumir qué está implementado y qué falta.

Antes de implementar un hito:

1. Explicar el enfoque propuesto.
2. Identificar archivos a crear o modificar.
3. Mencionar decisiones técnicas importantes y sus compromisos.
4. Pedir aprobación si implica una decisión arquitectónica significativa.

Al completar un hito:

1. Resumir la implementación.
2. Listar archivos modificados.
3. Reportar pruebas y resultados reales (sin afirmar verificaciones que no se ejecutaron).
4. Explicar los conceptos de RL involucrados.
5. Identificar la siguiente tarea lógica sin implementarla automáticamente.

**Priorizar siempre un sistema pequeño, funcional y medible sobre una implementación grande sin terminar.**

## 11. Estado del repositorio

- M1 implementado: paquete en capas `src/qb_sim/engine` (geometría, entidades, acciones híbridas `Move`/`Throw`, física) y `src/qb_sim/rules` (`Play`, rutas `go`/`slant`/`post`). Jugada determinista de pase con recepción, fase de portador, tackle de toque, intercepción, fuera de límites y límite de tiempo.
- M2 implementado: roster de 2 receptores + 2 defensivos con identidad por jugador, cobertura individual, QB humano con movimiento y lanzamiento *drag-and-throw* estilo resortera (jalar hacia atrás y soltar; previsualización parabólica punteada; `Throw(direction, power)` es la representación `(ángulo, fuerza)` que también usará el agente), pase con altura y arco parabólico (`arc_ratio`, `max_arc_height`, `catch_height`), precisión contextual con RNG local sembrable (`seed`, `reset(seed=...)`, `PlayState.pressure`; la desviación altera dirección y potencia, no el resultado), `src/qb_sim/ui` (`ViewTransform`, controles, `PlaySession` headless y app Pygame opcional con `pygame-ce`) y `scripts/accuracy_report.py` para calibración reproducible. 105 pruebas con pytest. Sin Gymnasium ni entrenamiento.
- M3 implementado: interfaz común de agente (`agents.base.BaseAgent` con `observe`/`act`), observaciones locales por rol, controladores de reglas (ruta, cobertura individual y QB heurístico), `Play` mecánico que recibe acciones por jugador, escenarios reproducibles (`default`, `pressured`, `deep`), registro/reproducción JSONL (`recording.py`) y métricas de referencia (`metrics.py` + `scripts/rule_baseline_report.py`).
- M4 implementado: entorno Gymnasium `QBPassingEnv` (`src/qb_sim/training`, registrado como `QBPassing-v0`), observaciones normalizadas de 27 valores, acción híbrida de 6 valores (`move`, `aim`, `power`, `trigger`) que se traduce a `Move`/`Throw`, recompensas configurables (`RewardConfig`), terminación/truncamiento e `info` diagnóstico; validado con el `check_env` de Gymnasium.
- M5 implementado: pipeline PPO con **warm start por imitación** del QB heurístico (`training/imitation.py`: recolección de demostraciones, BC ponderado y carga de pesos en el actor), fine-tuning conservador (lr 2e-5, `target_kl`, sin entropía, `log_std_init` bajo) y **pass rusher con sacks** (`PlayOutcome.SACK`, `PassRushController`, escenario `rushed`). Incluye checkpoints, reanudación, `config.json`, curvas CSV, evaluación reproducible por semillas fijas (`training/evaluation.py`, `scripts/evaluate_ppo.py`), análisis de comportamiento (`scripts/analyze_ppo.py`) y dashboard de entrenamiento en vivo para grabar (`ui/dashboard.py`, `scripts/train_dashboard.py`). 196 pruebas con pytest.
- Resultado de M5: con el roster `rushed` (2 CB cubriendo + 1 pass rusher dedicado `ru1`), PPO alcanza ~80% de completos y 0 intercepciones frente a 55% de la línea base de reglas (recompensa 1.1 vs 0.60); lanza a 1.0 s y aprende a lanzar en movimiento (100% de los pases con el QB a velocidad máxima) para escapar del rusher. La observación tiene layout fijo de 36 valores con slots por rol y flags de presencia (`training/observations.py`), de modo que un modelo puede jugar cualquier roster. Entrenamiento multi-escenario probado (`--scenarios`, `MultiScenarioEnv`): una sola política con observaciones cinemáticas no se estabiliza entre los 4 escenarios; queda como experimento pendiente con features de contexto de ruta o cabezas por escenario (M7). Modelo de producción: `runs/qb/play/baseline_v10`.
- M6 implementado: entorno multiagente PettingZoo `QBPassingMultiAgentEnv` (`training/multiagent.py`) con un agente por jugador (QB, receptores, cornerbacks y rusher), observaciones planas por rol (36 para el QB, 19 para receptores/defensivos en `training/role_observations.py`), acciones por rol (6 valores para el QB, 2 de movimiento para el resto), recompensas cero-sum en el resultado, episodios con reglas (`run_rule_episode`) y validación con `parallel_api_test` de PettingZoo. El modo humano y el entorno de un agente no cambian. 215 pruebas con pytest.
- M7 implementado (Stage A+B): motor de drive en `rules/drive.py` (3 downs, 10 yardas para el primero, LOS inicial aleatorio 25–50 con semilla, touchdown al cruzar la goal line, intercepción y turnover on downs como finales, límite de tiempo mapeado a coverage sack); `Play` acepta `line_of_scrimmage` y `goal_line` opcionales sin cambiar los defaults de M5/M6; catálogo de 8 rutas con sampling sembrado por jugada; observación v11 de 57 valores (36 base + one-hot de rutas 2×8 + down/distancia/yardas a goal) en `training/drive_observations.py`; entorno Gym `QBDriveEnv` (`QBDrive-v0`) donde el QB aprende por drive completo con el resto de los jugadores según reglas y rutas aleatorias; recompensa `DriveRewardConfig` (first down +0.3, TD +7, turnover on downs −0.5); BC de drive en `training/imitation.py`; evaluación con IC de Wilson (`training/drive_evaluation.py`, `scripts/evaluate_drive.py`). Se corrigió la convención de behavior cloning (targets crudos + BCE del trigger, consistente con `DiagGaussianDistribution` sin squash de SB3 2.9); `runs/qb/play/baseline_v10` permanece congelado como baseline jugada-suelta. `runs/qb/drive/random_routes_v11` (250k pasos, BC 800 épocas/lr 1e-4) supera la línea base de reglas en `rushed`: TD 30% vs 21%, puntos/drive 2.10 vs 1.47, primeros downs 1.24 vs 0.89, conversión de 3er down 43.3% vs 32.2%, 3er y larga 43.1% vs 32.2%, INT 37% vs 49%; la profundidad media en 3ra y larga queda en paridad (22.6 vs 22.7 yd) y ninguna política usa checkdown: la línea base de reglas ya lanza profundo, así que el criterio de profundidad del plan no diferencia; la mejora de v11 se concentra en conversión de 3er down (43.3% vs 32.2%), 3er y larga (43.1% vs 32.2%) e intercepciones (37% vs 49%). El dashboard de entrenamiento tiene modo drive (`scripts/train_dashboard.py --env drive --run-name qb/drive/random_routes_v11 --scenario rushed`) para ver drives con el QB entrenado, con navegación de checkpoints (←/→ anterior/siguiente, L volver al modo vivo, P pausa, 1/2/3 velocidad, R reiniciar). 297 pruebas con pytest.
- Pendiente de M7: entrenar por rol WR/CB/rusher a nivel jugada y composición multiagente (el WR necesitará one-hot de ruta en su observación para ejecutar rutas aleatorias); después MARL/coordinación y juego completo (M8).
- M7 Stage B implementado: el receptor corre al touchdown tras la atrapada y **fatiga del portador** (`carrier_fatigue_per_second=0.08`, piso 0.65, default apagado; calibración en `scripts/carrier_feasibility.py`: TD 1%→78% según la ventaja WR–CB); observación del WR de 31 valores (19 base + ruta 8 + yardas a goal + próximo waypoint 3) y de defensivos de 24 (19 base + down/distancia/campo); entornos `RolePlayEnv` (WR a nivel jugada) y `RoleDriveEnv` (CB/rusher en drives) con `training/role_context_observations.py`; adaptadores `ReceiverPolicy`/`DefenderPolicy`; entrenamiento por rol con oponentes congelados (`--env role_play|role_drive --learner --opponent`), BC + PPO y shaping potencial para defensivos (`DefenderRewardConfig`). Resultados: `runs/wr/route_progress_v1` supera la línea base de reglas en atrapadas (28% vs 18% con LOS aleatorio 25–50) y YAC 14.6 vs 21.7; `runs/cb/context_shaping_v1` fuerza más intercepciones (9.0% vs 8.8%) con separación/completos en paridad; el rusher no interactúa con el balón en vuelo (ni intercepta ni desvía; simplificación arcade al no modelarse bloqueo), `runs/ru/history/default_lr_v1` (lr 5e-6) queda en paridad de presión/arrival y mejora los TD permitidos (4% vs 6%) y los completos permitidos (7.6% vs 8.9%). Composición con `runs/qb/drive/trained_roles_v12` (fine-tune de v11 contra los roles entrenados) bajo la regla nueva: la línea base de reglas sube a TD 29%, mientras v12 queda en TD 11% con INT 9% vs 18% de reglas; `runs/qb/drive/from_scratch_v13` (entrenado desde cero en `stage_b`) no mejora (TD 9%). Pendiente re-tune del QB para la regla nueva. Se corrigieron la convención BC↔PPO (targets crudos), el start desplazado del receptor en drives (incluido el LOS aleatorio en `RolePlayEnv`), las features de ruta en cero al acarrear, la recompensa densa por avance del portador y el piso del spot en `Drive` para que la formación no salga del campo. **Nota de benchmark:** los números de evaluación de `runs/qb/play/baseline_v10` y `runs/qb/drive/random_routes_v11` (y sus líneas base de reglas) son dinámicas **pre-Stage-B**: el receptor de reglas ahora corre al touchdown tras la atrapada y `stage_b` activa la fatiga del portador, por lo que los benchmarks con reglas cambiaron; esos números se conservan tal como se midieron. Pendiente: PPO del rusher y MARL/M8. El dashboard de entrenamiento tiene modos por rol (`scripts/train_dashboard.py --env role_play|role_drive --learner wr1 --opponent qb=runs/...`) para ver al WR/CB/rusher entrenados contra oponentes congelados, con el jugador aprendido resaltado y navegación de checkpoints (←/→, L vivo); muestra `modo: demo` o `entrenando` y `--train` no pisa un run existente sin `--force`. Además hay modo **composición** (`--env composition`) que corre todos los modelos entrenados juntos (QB v12 + WR v1 + CB v1 + RU bc por defecto, descubiertos de `runs/`), con menú `M` para editar en vivo los parámetros numéricos de `SimulationConfig`/`DriveConfig` (velocidades, arranques del rusher, fatiga, tackle, LOS, etc.) y menú `N` para cambiar el modelo por rol; los cambios reinician los paneles al instante. El dashboard usa un tema oscuro data-dense (`ui/theme.py`: paleta azul/ámbar, tarjetas de sección, gráfica con área y tipografías sans/mono con fallback, cabecera con run/modo/modelo). 458 pruebas con pytest.
- M7 Stage B2 implementado (adaptación a configs + throughput): `train.py` honra `--n-envs` con `SubprocVecEnv` (~4.3k fps con 8 envs, ~3x; cuello IPC/stepping) y expone `--n-steps`/`--batch-size`/`--learning-rate` (con `n_envs=8` usar `--n-steps 128` para mantener ~1024 pasos/update: 250k con n_steps=1024 dan solo ~30 updates); observaciones con capacidades asimétricas (QB 61, WR 35, defensivos 27 con `DriveState.last_throw_distance`); maestros de reglas conscientes del rango (`build_route(max_depth_yards)`, QB solo lanza alcanzables); randomización uniforme de 6 escalares (velocidades, `throw_max_range`, sigma) con curriculum en dos fases, semilla de config separada del layout e inyección del config activo en oponentes (`set_config`); especialistas quedan como números de referencia (layouts 57/31/24 incompatibles con 61/35/27). Resultados: roles generalizados (`runs/wr/adaptation_g1`, `runs/cb/adaptation_g1`, `runs/ru/adaptation_g1`) en paridad con reglas en la config base (WR: completos de equipo 36% vs 35%; CB: completos permitidos 22.5% vs 22.5%, INT 12.3% vs 13.8%; rusher: 27.0% vs 27.5%, arrival 2.08 vs 2.07). `runs/qb/drive/adaptation_g1` (1M pasos, BC randomizado) **no explota aún la ventaja de velocidad**: en la matriz gap × rango las reglas logran TD 1%→82% según el gap (el comportamiento pedido), mientras el QB aprendido queda en 0-13% con 15-100% de scramble; el BC randomizado enseñó a moverse siempre y PPO no lo corrigió (problema abierto de credit assignment/matchup). Siguiente: MARL (IPPO con snapshots) y mejorar la explotación del matchup del QB (más updates, shaping o BC por arquetipos).
- Pendiente del roadmap: MARL y coordinación multiagente (M7, parcial) y juego completo (M8). Los receptores y defensivos siguen usando controladores de reglas fuera de las políticas entrenadas.
- `docs/superpowers/plans/2026-10-01-qb-simulation-core.md` es histórico; los planes vigentes son `2026-10-01-m1-simulation-engine.md`, `2026-10-01-m2-human-playable-prototype.md`, `2026-10-01-m3-rule-based-controllers.md`, `2026-10-01-m4-gymnasium-environment.md`, `2026-10-01-m5-ppo-training.md` y `2026-10-01-m6-multiagent-environment.md`.

## 12. Decisiones abiertas (requieren aprobación antes de implementar)

No hay decisiones abiertas.

Resueltas en M2: alcance del primer prototipo jugable (QB humano + 2 receptores + 2 defensivos), esquema de acciones híbrido (`Move` y `Throw(direction, power)` como `(ángulo, fuerza)`), regla de tackle de toque y fidelidad del pase con altura y arco parabólico. Resuelta en M3: formato de registro/reproducción (JSONL legible con semilla, configuración, acciones por tick y resultado). Resuelta en M6: interfaz multiagente = PettingZoo `ParallelEnv` con observaciones y acciones por rol.
