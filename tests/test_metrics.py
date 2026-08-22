import numpy as np

from ddi.metrics import (
    bootstrap_metric_intervals,
    conformal_set_metrics,
    decision_cost,
    selective_metrics,
)


def test_decision_cost_counts_false_actions_and_deferrals() -> None:
    y = np.array([0, 0, 1, 1])
    decisions = np.array([0, 1, 0, -1])
    cost = decision_cost(y, decisions, 1.0, 20.0, 0.5)
    assert cost == (1.0 + 20.0 + 0.5) / 4


def test_selective_metrics_report_automation_coverage() -> None:
    y = np.array([0, 0, 1, 1])
    decisions = np.array([0, -1, 1, -1])
    probabilities = np.array([0.1, 0.4, 0.9, 0.6])
    metrics = selective_metrics(y, decisions, probabilities, 1.0, 20.0, 0.5)
    assert metrics["automation_coverage"] == 0.5
    assert metrics["selective_risk"] == 0.0
    assert metrics["system_accuracy_assuming_correct_review"] == 1.0


def test_conformal_set_metrics_distinguish_validity_from_automation() -> None:
    y = np.array([0, 1, 0, 1])
    prediction_sets = np.array(
        [[True, False], [True, True], [False, True], [False, True]]
    )
    metrics = conformal_set_metrics(y, prediction_sets)
    assert metrics["empirical_set_coverage"] == 0.75
    assert metrics["empirical_class_zero_coverage"] == 0.5
    assert metrics["empirical_class_one_coverage"] == 1.0
    assert metrics["mean_prediction_set_size"] == 1.25
    assert metrics["singleton_rate"] == 0.75
    assert metrics["empty_set_rate"] == 0.0
    assert metrics["ambiguous_set_rate"] == 0.25


def test_bootstrap_table_uses_observed_point_estimates() -> None:
    y = np.array([0, 0, 1, 1])
    proposed = np.array([0, -1, 1, -1])
    sets = np.array(
        [[True, False], [False, True], [False, True], [False, True]]
    )
    baseline = np.array([0, 0, 1, 0])
    probabilities = np.array([0.1, 0.4, 0.9, 0.6])
    intervals = bootstrap_metric_intervals(
        y,
        proposed,
        sets,
        baseline,
        probabilities,
        1.0,
        20.0,
        0.5,
        repetitions=20,
        seed=7,
    ).set_index("metric")
    assert intervals.loc["automation_coverage", "estimate"] == 0.5
    assert intervals.loc["empirical_set_coverage", "estimate"] == 0.75
    assert intervals.loc["proposed_mean_cost", "estimate"] == 0.25
