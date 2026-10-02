from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.evaluation import RandomPolicy, RuleQuarterbackPolicy, evaluate


def test_evaluation_is_reproducible_with_fixed_seeds() -> None:
    first = evaluate(RandomPolicy(seed=0), "default", range(10))
    second = evaluate(RandomPolicy(seed=0), "default", range(10))

    assert first == second


def test_metrics_include_the_required_keys() -> None:
    summary = evaluate(RandomPolicy(seed=1), "default", range(5))

    expected = {
        "episodes",
        "completed_pct",
        "interception_pct",
        "incomplete_pct",
        "yards_per_attempt",
        "mean_reward",
        "average_duration",
        "throw_frequency",
        "move_frequency",
        "average_throw_time",
    }
    assert expected <= set(summary)
    assert summary["episodes"] == 5


def test_rule_quarterback_completes_more_than_random() -> None:
    config = get_scenario("default").config
    seeds = range(50)

    rules = evaluate(RuleQuarterbackPolicy(config), "default", seeds)
    random = evaluate(RandomPolicy(seed=7), "default", seeds)

    assert rules["completed_pct"] > random["completed_pct"]
    assert rules["throw_frequency"] > 0.0
