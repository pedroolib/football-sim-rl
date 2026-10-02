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

SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario for scenario in (DEFAULT_SCENARIO, PRESSURED_SCENARIO, DEEP_SCENARIO)
}


def get_scenario(name: str) -> Scenario:
    """Return a named scenario or raise ``ValueError``."""
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(f"unknown scenario: {name!r}") from error
