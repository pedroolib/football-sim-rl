"""Headless orchestration: rule-based teams and reproducible series."""

from collections.abc import Mapping

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.rule_controllers import (
    HeuristicQuarterback,
    ManCoverageController,
    PassRushController,
    ReceiverRouteController,
)
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayOutcome, PlayState
from qb_sim.recording import PlayRecorder
from qb_sim.rules.play import Play
from qb_sim.rules.routes import RouteName, build_route
from qb_sim.rules.scenarios import DEFAULT_SCENARIO, Scenario


def build_rule_agents(
    play: Play,
    routes: tuple[RouteName, ...],
    config: SimulationConfig | None = None,
    include_quarterback: bool = True,
) -> dict[str, BaseAgent]:
    """Create one rule-based agent per player in roster order.

    Cornerbacks cover receivers in order; roster rushers rush the quarterback.
    """
    settings = config if config is not None else play.config
    agents: dict[str, BaseAgent] = {}
    if include_quarterback:
        agents["qb"] = HeuristicQuarterback("qb", settings)
    for index, player_id in enumerate(play.receiver_ids):
        start = play.state.player(player_id).position
        route = build_route(routes[index], start, settings.field, settings)
        agents[player_id] = ReceiverRouteController(
            player_id,
            route,
            settings.receiver_speed,
            settings.dt,
        )
    rusher_ids = set(play.rusher_ids)
    coverage_index = 0
    for player_id in play.defender_ids:
        if player_id in rusher_ids:
            agents[player_id] = PassRushController(
                player_id,
                settings.defender_speed,
                settings.dt,
            )
            continue
        assignment = play.receiver_ids[coverage_index % len(play.receiver_ids)]
        coverage_index += 1
        agents[player_id] = ManCoverageController(
            player_id,
            settings.defender_speed,
            assignment,
            settings.dt,
        )
    return agents


def run_play(
    scenario: Scenario | None = None,
    controllers: Mapping[str, BaseAgent] | None = None,
    seed: int | None = None,
    recorder: PlayRecorder | None = None,
) -> PlayState:
    """Run one full play headless and return its final state."""
    chosen = scenario if scenario is not None else DEFAULT_SCENARIO
    play_seed = chosen.seed if seed is None else seed
    play = Play(chosen.config, seed=play_seed)
    agents = (
        dict(controllers)
        if controllers is not None
        else build_rule_agents(play, chosen.receiver_routes)
    )
    for agent in agents.values():
        agent.reset()
    if recorder is not None:
        recorder.record_header(
            seed=play_seed,
            scenario=chosen.name,
            config=chosen.config,
            routes=chosen.receiver_routes,
        )
    state = play.state
    while state.outcome is PlayOutcome.IN_PROGRESS:
        actions: dict[str, Move | Throw] = {}
        for player_id, agent in agents.items():
            action = agent.decide(state)
            if action is not None:
                actions[player_id] = action
        if recorder is not None:
            recorder.record_tick(actions)
        state = play.step(actions)
    if recorder is not None:
        recorder.record_result(state)
    return state


def run_series(
    scenario: Scenario | None = None,
    count: int = 10,
    seed_start: int = 0,
) -> list[PlayState]:
    """Run ``count`` seeded plays and return their final states."""
    if count <= 0:
        raise ValueError("count must be positive")
    chosen = scenario if scenario is not None else DEFAULT_SCENARIO
    return [run_play(chosen, seed=seed_start + index) for index in range(count)]
