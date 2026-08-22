from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from ddi.metrics import risk_coverage_curve


def _save(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_reliability(
    y_true: np.ndarray,
    uncalibrated: np.ndarray,
    calibrated: np.ndarray,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    for probabilities, label, marker in [
        (uncalibrated, "Uncalibrated knowledge XGBoost", "o"),
        (calibrated, "Platt-calibrated knowledge XGBoost", "s"),
    ]:
        observed, predicted = calibration_curve(
            y_true, probabilities, n_bins=10, strategy="quantile"
        )
        axis.plot(predicted, observed, marker=marker, linewidth=2, label=label)
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect calibration")
    axis.set(xlabel="Mean predicted failure probability", ylabel="Observed failure rate")
    axis.set_title("Reliability diagram")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save(figure, destination)


def plot_risk_coverage(
    y_true: np.ndarray, probabilities: np.ndarray, destination: Path
) -> None:
    curve = risk_coverage_curve(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    axis.plot(curve["coverage"], curve["selective_risk"], linewidth=2.2)
    axis.set(
        xlabel="Automatic coverage (fraction not deferred)",
        ylabel="Error rate among automatic predictions",
        title="Risk–coverage trade-off",
    )
    axis.grid(alpha=0.25)
    _save(figure, destination)


def plot_alpha_tradeoff(alpha_frame: pd.DataFrame, destination: Path) -> None:
    grouped = alpha_frame.groupby("alpha", as_index=False)[
        ["coverage", "selective_risk", "mean_decision_cost"]
    ].mean()
    figure, left_axis = plt.subplots(figsize=(7.2, 5.2))
    right_axis = left_axis.twinx()
    left_axis.plot(
        grouped["alpha"], grouped["coverage"], marker="o", label="Coverage"
    )
    left_axis.plot(
        grouped["alpha"], grouped["selective_risk"], marker="s", label="Selective risk"
    )
    right_axis.plot(
        grouped["alpha"],
        grouped["mean_decision_cost"],
        marker="^",
        color="tab:red",
        label="Decision cost",
    )
    left_axis.set(xlabel="Conformal alpha", ylabel="Rate")
    right_axis.set_ylabel("Mean decision cost", color="tab:red")
    left_axis.set_title("Conformal error level versus autonomy, risk, and cost")
    left_axis.grid(alpha=0.25)
    lines = left_axis.get_lines() + right_axis.get_lines()
    left_axis.legend(lines, [line.get_label() for line in lines], fontsize=8)
    _save(figure, destination)


def plot_feature_importance(
    feature_names: list[str], importances: np.ndarray, destination: Path
) -> None:
    frame = pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(importances, dtype=float)}
    ).sort_values("importance", ascending=False).head(12)
    frame = frame.sort_values("importance")
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.barh(frame["feature"], frame["importance"], color="tab:blue")
    axis.set(xlabel="XGBoost feature importance", title="Top model features")
    axis.grid(axis="x", alpha=0.25)
    _save(figure, destination)

