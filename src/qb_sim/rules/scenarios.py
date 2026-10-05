"""Named, reproducible play scenarios."""

from dataclasses import dataclass, replace

from qb_sim.config import SimulationConfig
from qb_sim.rules.routes import RouteName


@dataclass(frozen=True, slots=True)
class Scenario:
    """A reproducible play setup: configuration, routes, and base seed."""

    name: str
    config: SimulationConfig
    receiver_routes: tuple[RouteName, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.receiver_routes) != len(self.config.receiver_starts):
            raise ValueError("a route is required for each receiver")


DEFAULT_SCENARIO = Scenario(
    "default",
    SimulationConfig(),
    (RouteName.POST, RouteName.SLANT),
)
PRESSURED_SCENARIO = Scenario(
    "pressured",
    replace(SimulationConfig(), defender_starts=((18.5, 26.65), (0.0, 0.0))),
    (RouteName.POST, RouteName.SLANT),
)
DEEP_SCENARIO = Scenario(
    "deep",
    SimulationConfig(),
    (RouteName.GO, RouteName.POST),
)
RUSHED_SCENARIO = Scenario(
    "rushed",
    replace(SimulationConfig(), rusher_starts=((30.0, 29.0),)),
    (RouteName.POST, RouteName.SLANT),
)
FAST_COVERAGE_SCENARIO = Scenario(
    "fast_coverage",
    replace(
        SimulationConfig(),
        rusher_starts=((30.0, 29.0),),
        receiver_speed=7.5,
        defender_speed=7.0,
    ),
    (RouteName.POST, RouteName.SLANT),
)
SLOW_RUSH_SCENARIO = Scenario(
    "slow_rush",
    replace(
        SimulationConfig(),
        rusher_starts=((30.0, 29.0),),
        rusher_speed=5.0,
    ),
    (RouteName.POST, RouteName.SLANT),
)
FAST_RUSH_SCENARIO = Scenario(
    "fast_rush",
    replace(
        SimulationConfig(),
        rusher_starts=((30.0, 29.0),),
        rusher_speed=8.0,
    ),
    (RouteName.POST, RouteName.SLANT),
)
MIXED_SCENARIO = Scenario(
    "mixed",
    replace(
        SimulationConfig(),
        rusher_starts=((30.0, 29.0),),
        receiver_speed=7.5,
        defender_speed=6.8,
        rusher_speed=6.0,
    ),
    (RouteName.POST, RouteName.SLANT),
)

STAGE_B_CONFIG = replace(
    SimulationConfig(),
    rusher_starts=((30.0, 29.0),),
    carrier_fatigue_per_second=0.12,
    carrier_min_speed_factor=0.65,
    qb_snap_depth_yards=2.0,
    qb_run_speed_factor=0.6,
)
STAGE_B_SCENARIO = Scenario(
    "stage_b",
    STAGE_B_CONFIG,
    (RouteName.POST, RouteName.SLANT),
)

SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        DEFAULT_SCENARIO,
        PRESSURED_SCENARIO,
        DEEP_SCENARIO,
        RUSHED_SCENARIO,
        FAST_COVERAGE_SCENARIO,
        SLOW_RUSH_SCENARIO,
        FAST_RUSH_SCENARIO,
        MIXED_SCENARIO,
        STAGE_B_SCENARIO,
    )
}


def get_scenario(name: str) -> Scenario:
    """Return a named scenario or raise ``ValueError``."""
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(f"unknown scenario: {name!r}") from error
