from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            in_bin = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            in_bin = (p >= edges[index]) & (p < edges[index + 1])
        if not np.any(in_bin):
            continue
        confidence = float(np.mean(p[in_bin]))
        observed = float(np.mean(y[in_bin]))
        error += float(np.sum(in_bin)) / total * abs(confidence - observed)
    return float(error)


def predictive_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = (p >= threshold).astype(int)
    negative_mask = y == 0
    specificity = float(np.mean(predicted[negative_mask] == 0))
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }


def decision_cost(
    y_true: np.ndarray,
    decisions: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    deferral_cost: float,
) -> float:
    """Mean cost; -1 means defer to an assumed-correct human reviewer."""

    y = np.asarray(y_true, dtype=int)
    action = np.asarray(decisions, dtype=int)
    false_positive = (action == 1) & (y == 0)
    false_negative = (action == 0) & (y == 1)
    deferred = action == -1
    total = (
        false_positive_cost * np.sum(false_positive)
        + false_negative_cost * np.sum(false_negative)
        + deferral_cost * np.sum(deferred)
    )
    return float(total / len(y))


def selective_metrics(
    y_true: np.ndarray,
    decisions: np.ndarray,
    reference_probabilities: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    deferral_cost: float,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    action = np.asarray(decisions, dtype=int)
    accepted = action != -1
    deferred = ~accepted
    accepted_count = int(np.sum(accepted))
    accepted_errors = int(np.sum(action[accepted] != y[accepted])) if accepted_count else 0
    reference_prediction = (np.asarray(reference_probabilities) >= 0.5).astype(int)
    reference_errors = reference_prediction != y
    error_capture = (
        float(np.sum(reference_errors & deferred) / np.sum(reference_errors))
        if np.any(reference_errors)
        else 0.0
    )
    positive_count = max(1, int(np.sum(y == 1)))
    automatic_true_positives = int(np.sum((action == 1) & (y == 1)))
    return {
        "coverage": float(np.mean(accepted)),
        "deferral_rate": float(np.mean(deferred)),
        "selective_accuracy": (
            float(np.mean(action[accepted] == y[accepted])) if accepted_count else float("nan")
        ),
        "selective_risk": (
            float(accepted_errors / accepted_count) if accepted_count else float("nan")
        ),
        "system_accuracy_assuming_correct_review": float(1.0 - accepted_errors / len(y)),
        "automatic_failure_recall": float(automatic_true_positives / positive_count),
        "reference_errors_deferred": error_capture,
        "mean_decision_cost": decision_cost(
            y,
            action,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            deferral_cost=deferral_cost,
        ),
    }


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    proposed_decisions: np.ndarray,
    baseline_decisions: np.ndarray,
    reference_probabilities: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    deferral_cost: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    proposed = np.asarray(proposed_decisions, dtype=int)
    baseline = np.asarray(baseline_decisions, dtype=int)
    p = np.asarray(reference_probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "coverage": [],
        "selective_risk": [],
        "system_accuracy_assuming_correct_review": [],
        "proposed_mean_cost": [],
        "baseline_mean_cost": [],
        "cost_reduction_vs_baseline": [],
    }
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), size=len(y))
        proposed_metrics = selective_metrics(
            y[idx],
            proposed[idx],
            p[idx],
            false_positive_cost,
            false_negative_cost,
            deferral_cost,
        )
        proposed_cost = proposed_metrics["mean_decision_cost"]
        baseline_cost = decision_cost(
            y[idx],
            baseline[idx],
            false_positive_cost,
            false_negative_cost,
            deferral_cost,
        )
        samples["coverage"].append(proposed_metrics["coverage"])
        samples["selective_risk"].append(proposed_metrics["selective_risk"])
        samples["system_accuracy_assuming_correct_review"].append(
            proposed_metrics["system_accuracy_assuming_correct_review"]
        )
        samples["proposed_mean_cost"].append(proposed_cost)
        samples["baseline_mean_cost"].append(baseline_cost)
        samples["cost_reduction_vs_baseline"].append(baseline_cost - proposed_cost)

    rows: list[dict[str, float | str]] = []
    for metric, values in samples.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "estimate": float(np.nanmean(array)),
                "ci_lower_95": float(np.nanpercentile(array, 2.5)),
                "ci_upper_95": float(np.nanpercentile(array, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def risk_coverage_curve(
    y_true: np.ndarray, probabilities: np.ndarray, points: int = 80
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    prediction = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    thresholds = np.linspace(0.5, min(0.999, float(np.max(confidence))), points)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        if not np.any(accepted):
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(np.mean(accepted)),
                "selective_risk": float(np.mean(prediction[accepted] != y[accepted])),
            }
        )
    return pd.DataFrame(rows).sort_values("coverage")


def summarize_across_seeds(
    frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    grouped = frame.groupby(group_columns, dropna=False)[numeric].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()

