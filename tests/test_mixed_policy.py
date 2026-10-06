import pytest

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.rule_controllers import HeuristicQuarterback
from qb_sim.engine.entities import Role
from qb_sim.training import policy as policy_module
from qb_sim.training.policy import MixedPolicy


class _RecordingCallable:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple] = []

    def __call__(self, state, *args, **kwargs):
        self.calls.append((state, args, kwargs))
        return self.name


class _Configurable:
    def __init__(self) -> None:
        self.configs: list = []

    def set_config(self, config) -> None:
        self.configs.append(config)


class _FakeAgent(BaseAgent):
    role = Role.QUARTERBACK

    def __init__(self, name: str = "qb") -> None:
        super().__init__(name)
        self.decisions: list = []

    def observe(self, state):
        return state

    def act(self, observation):
        return "act"

    def decide(self, state):
        self.decisions.append(state)
        return "decided"


def _import_train():
    pytest.importorskip("stable_baselines3")
    from qb_sim.training import train as train_module

    return train_module


def test_mixed_policy_requires_at_least_one_policy() -> None:
    with pytest.raises(ValueError, match="al menos"):
        MixedPolicy(())


def test_mixed_policy_keeps_one_policy_for_the_whole_play() -> None:
    first = _RecordingCallable("first")
    second = _RecordingCallable("second")
    mixed = MixedPolicy((first, second), seed=2)
    state = object()

    results = [mixed(state) for _ in range(3)]

    chosen, other = (first, second) if first.calls else (second, first)
    assert results == [chosen.name] * 3
    assert len(chosen.calls) == 3
    assert other.calls == []


def test_mixed_policy_resamples_on_a_new_play_state() -> None:
    first = _RecordingCallable("first")
    second = _RecordingCallable("second")
    mixed = MixedPolicy((first, second), seed=0)

    results = [mixed(object()) for _ in range(20)]

    assert set(results) == {"first", "second"}
    assert len(first.calls) + len(second.calls) == 20


def test_mixed_policy_sampling_is_deterministic_for_a_seed() -> None:
    def sequence(seed: int) -> list[str]:
        first = _RecordingCallable("first")
        second = _RecordingCallable("second")
        mixed = MixedPolicy((first, second), seed=seed)
        return [mixed(object()) for _ in range(20)]

    assert sequence(7) == sequence(7)


def test_mixed_policy_dispatches_a_base_agent_through_decide() -> None:
    agent = _FakeAgent()
    mixed = MixedPolicy((agent,), seed=0)
    state = object()

    assert mixed(state) == "decided"
    assert agent.decisions == [state]


def test_mixed_policy_forwards_arguments_to_a_callable() -> None:
    policy = _RecordingCallable("policy")
    mixed = MixedPolicy((policy,), seed=0)
    state = object()

    mixed(state, "route", 3, goal_line=100.0)

    assert policy.calls == [(state, ("route", 3), {"goal_line": 100.0})]


def test_mixed_policy_forwards_set_config_to_every_supported_policy() -> None:
    first = _Configurable()
    second = _Configurable()
    agent = _FakeAgent()
    mixed = MixedPolicy((first, agent, second), seed=0)
    config = object()

    mixed.set_config(config)

    assert first.configs == [config]
    assert second.configs == [config]


def test_mixed_policy_reset_clears_the_cached_selection() -> None:
    first = _RecordingCallable("first")
    second = _RecordingCallable("second")
    mixed = MixedPolicy((first, second), seed=2)
    state = object()

    before = mixed(state)
    mixed.reset()
    after = mixed(state)

    assert before != after


def test_build_opponents_wraps_mixed_parts(monkeypatch) -> None:
    train_module = _import_train()
    learned = _RecordingCallable("learned")
    monkeypatch.setattr(policy_module, "load_drive_policy", lambda path, scenario: learned)

    opponents = train_module.build_opponents(
        "role_drive",
        "stage_b",
        (("qb", "rules|runs/qb/drive/ep_v1/best/best_model.zip"),),
    )

    mixed = opponents["qb"]
    assert isinstance(mixed, MixedPolicy)
    assert learned in mixed.policies
    assert any(isinstance(policy, HeuristicQuarterback) for policy in mixed.policies)
    assert len(mixed.policies) == 2


def test_build_opponents_keeps_a_single_path_unwrapped(monkeypatch) -> None:
    train_module = _import_train()
    learned = _RecordingCallable("learned")
    monkeypatch.setattr(policy_module, "load_drive_policy", lambda path, scenario: learned)

    opponents = train_module.build_opponents(
        "role_drive",
        "stage_b",
        (("qb", "runs/qb/drive/ep_v1/best/best_model.zip"),),
    )

    assert opponents["qb"] is learned


def test_build_opponents_builds_rule_agents_for_non_qb_ids() -> None:
    train_module = _import_train()

    opponents = train_module.build_opponents("role_drive", "stage_b", (("cb1", "rules"),))

    assert isinstance(opponents["cb1"], HeuristicQuarterback)


def test_build_opponents_rejects_empty_mixed_segments() -> None:
    train_module = _import_train()

    with pytest.raises(ValueError, match="vacía"):
        train_module.build_opponents("role_drive", "stage_b", (("qb", "rules|"),))
