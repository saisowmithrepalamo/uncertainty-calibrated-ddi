import numpy as np

from ddi.metrics import decision_cost, selective_metrics


def test_decision_cost_counts_false_actions_and_deferrals() -> None:
    y = np.array([0, 0, 1, 1])
    decisions = np.array([0, 1, 0, -1])
    cost = decision_cost(y, decisions, 1.0, 20.0, 0.5)
    assert cost == (1.0 + 20.0 + 0.5) / 4


def test_selective_metrics_report_coverage() -> None:
    y = np.array([0, 0, 1, 1])
    decisions = np.array([0, -1, 1, -1])
    probabilities = np.array([0.1, 0.4, 0.9, 0.6])
    metrics = selective_metrics(y, decisions, probabilities, 1.0, 20.0, 0.5)
    assert metrics["coverage"] == 0.5
    assert metrics["selective_risk"] == 0.0
    assert metrics["system_accuracy_assuming_correct_review"] == 1.0

